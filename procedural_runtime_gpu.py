from __future__ import annotations

from pathlib import Path
from collections import Counter
from functools import lru_cache
import math

import numpy as np
import torch
import xxhash

from procedural_runtime_v5 import *


@lru_cache(maxsize=250_000)
def _h64_signed(text: str) -> int:
    # Map unsigned xxHash64 order to signed int64 order so torch.searchsorted can
    # use native CUDA int64 tensors while retaining all 64 hash bits.
    u = xxhash.xxh64_intdigest(text, seed=0xBACA2026)
    return int(u - (1 << 63))


class GpuBagacoSurfaceScorer(BagacoSurfaceScorer):
    """Non-neural Bagaço scorer accelerated with PyTorch CUDA tensor operations.

    No neural network, weights, gradients or backpropagation are used. PyTorch is
    only the CUDA runtime for sorted table lookup, logarithms and reductions.
    """
    GPU_TABLES = ("tokens", "p2", "p3", "p4", "p5")

    def __init__(self, root: str | Path, use_hot: bool = False,
                 device: int = 0, memory_limit_mb: int = 4608,
                 cache_dir: str | Path | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponivel no PyTorch")
        self.device_id = int(device)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.memory_limit_mb = int(memory_limit_mb)
        torch.cuda.set_device(self.device)
        props = torch.cuda.get_device_properties(self.device)
        if self.memory_limit_mb > 0:
            frac = min(0.95, max(0.10, (self.memory_limit_mb * 1024**2) / props.total_memory))
            torch.cuda.set_per_process_memory_fraction(frac, self.device)
        super().__init__(root, use_hot=use_hot)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(root) / "gpu_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gpu = {}
        self._build_or_load_gpu_index()
        self.batch_language_support([["de", "acordo", "com"], ["no", "caso", "de"]])
        torch.cuda.synchronize(self.device)

    def _cache_file(self) -> Path:
        tag = "hot" if self.use_hot else "quality"
        return self.cache_dir / f"bagaco_{tag}_xxh64_signed_v3.npz"

    @staticmethod
    def _hash_array(strings):
        return np.fromiter((_h64_signed(x) for x in strings), dtype=np.int64, count=len(strings))

    def _build_or_load_gpu_index(self):
        p = self._cache_file()
        arrays = None
        if p.exists():
            try:
                z = np.load(p, allow_pickle=False)
                if int(z["format_version"][0]) == 3:
                    arrays = {name: (z[f"{name}_h"], z[f"{name}_c"]) for name in self.GPU_TABLES}
            except Exception:
                arrays = None
        if arrays is None:
            arrays = {}
            save = {"format_version": np.asarray([3], dtype=np.int32)}
            for name in self.GPU_TABLES:
                tab = self.tables.get(name, {})
                h = np.fromiter((_h64_signed(k) for k in tab.keys()), dtype=np.int64, count=len(tab))
                c = np.fromiter((int(v) for v in tab.values()), dtype=np.float64, count=len(tab))
                order = np.argsort(h, kind="stable")
                h = h[order]; c = c[order]
                if len(h) > 1 and np.any(h[1:] == h[:-1]):
                    raise RuntimeError(f"colisao xxHash64 detectada em {name}; indice GPU rejeitado")
                arrays[name] = (h, c)
                save[f"{name}_h"] = h; save[f"{name}_c"] = c
            np.savez(p, **save)
        for name, (h, c) in arrays.items():
            self._gpu[name] = (
                torch.as_tensor(h, dtype=torch.int64, device=self.device),
                torch.as_tensor(c, dtype=torch.float64, device=self.device),
            )
        torch.cuda.synchronize(self.device)

    def _lookup(self, table: str, hashes_np: np.ndarray):
        keys, vals = self._gpu[table]
        if hashes_np.size == 0:
            return torch.empty((0,), dtype=torch.float64, device=self.device)
        q = torch.as_tensor(hashes_np, dtype=torch.int64, device=self.device)
        idx = torch.searchsorted(keys, q)
        clipped = torch.clamp(idx, max=max(0, keys.numel()-1))
        ok = (idx < keys.numel()) & (keys[clipped] == q)
        return torch.where(ok, vals[clipped], torch.zeros_like(vals[clipped]))

    def batch_language_support(self, batch_words, max_order: int = 4, slot_aware: bool = True):
        B = len(batch_words)
        if B == 0:
            return np.empty(0), np.empty(0)
        total = torch.zeros(B, dtype=torch.float64, device=self.device)
        weight = torch.zeros(B, dtype=torch.float64, device=self.device)

        tok_s=[]; tok_i=[]; valids=[]
        for bi, ws in enumerate(batch_words):
            valid=[not (slot_aware and self.is_slot(w)) for w in ws]
            valids.append(valid)
            for j,w in enumerate(ws):
                if valid[j]: tok_s.append(w); tok_i.append(bi)
        if tok_s:
            cnt=self._lookup("tokens",self._hash_array(tok_s))
            ids=torch.as_tensor(np.asarray(tok_i,dtype=np.int64),device=self.device)
            contrib=0.10*torch.log((cnt+0.5)/(self.total_tok+0.5*self.V))
            total += torch.bincount(ids,weights=contrib,minlength=B)
            weight += torch.bincount(ids,weights=torch.full_like(contrib,0.10),minlength=B)

        nweights={2:1.0,3:0.75,4:0.45,5:0.25}
        support_hit=torch.zeros(B,dtype=torch.float64,device=self.device)
        support_den=torch.zeros(B,dtype=torch.float64,device=self.device)
        for n in range(2,min(int(max_order),5)+1):
            grams=[];prefs=[];cids=[]
            for bi,ws in enumerate(batch_words):
                valid=valids[bi]
                for end in range(n-1,len(ws)):
                    start=end-n+1
                    if not all(valid[start:end+1]): continue
                    grams.append("\t".join(ws[start:end+1]))
                    prefs.append("\t".join(ws[start:end]))
                    cids.append(bi)
            if not grams: continue
            ids=torch.as_tensor(np.asarray(cids,dtype=np.int64),device=self.device)
            gc=self._lookup(f"p{n}",self._hash_array(grams))
            pc=self._lookup("tokens" if n==2 else f"p{n-1}",self._hash_array(prefs))
            alpha=0.05 if n==2 else 0.02
            fallback=math.log(alpha/(self.total_tok+alpha*self.V))
            lp=torch.where(pc>0,torch.log((gc+alpha)/(pc+alpha*self.V)),torch.full_like(gc,fallback))
            wgt=nweights[n]
            total += torch.bincount(ids,weights=wgt*lp,minlength=B)
            weight += torch.bincount(ids,weights=torch.full_like(lp,wgt),minlength=B)
            if n==3:
                support_hit += torch.bincount(ids,weights=(gc>0).to(torch.float64),minlength=B)
                support_den += torch.bincount(ids,weights=torch.ones_like(gc),minlength=B)

        lang=total/torch.clamp(weight,min=1e-9)
        support=support_hit/torch.clamp(support_den,min=1.0)
        torch.cuda.synchronize(self.device)
        return lang.detach().cpu().numpy(), support.detach().cpu().numpy()

    def score_tokens(self, words, max_order=4, slot_aware=True):
        lang,_=self.batch_language_support([list(words)],max_order=max_order,slot_aware=slot_aware)
        return float(lang[0])

    def supported_fraction(self,text,order=3,slot_aware=True):
        if int(order)!=3:
            return super().supported_fraction(text,order=order,slot_aware=slot_aware)
        _,support=self.batch_language_support([self.tokenize(text)],max_order=3,slot_aware=slot_aware)
        return float(support[0])

    def gpu_status(self):
        props=torch.cuda.get_device_properties(self.device)
        free_b,total_b=torch.cuda.mem_get_info(self.device)
        return {
            "backend":"pytorch-cuda-tensors",
            "neural_network":False,
            "gradients":False,
            "device":self.device_id,
            "name":props.name,
            "cuda":torch.version.cuda,
            "torch":torch.__version__,
            "vram_total_mb":round(total_b/2**20,1),
            "vram_free_mb":round(free_b/2**20,1),
            "torch_allocated_mb":round(torch.cuda.memory_allocated(self.device)/2**20,1),
            "torch_reserved_mb":round(torch.cuda.memory_reserved(self.device)/2**20,1),
            "memory_limit_mb":self.memory_limit_mb,
            "tables_on_gpu":list(self._gpu),
            "cache":str(self._cache_file()),
        }


class LearnedSurfaceSelectorGPU(LearnedSurfaceSelectorV7):
    """V9 selector that sends all cache-miss candidate language features as one CUDA batch."""
    def _static_features_many(self,texts,paragraph_first):
        rows=[None]*len(texts);misses=[];miss_words=[];miss_keys=[]
        for idx,text in enumerate(texts):
            ws=self.s.tokenize(text)
            key=(tuple("__slot__" if self.s.is_slot(w) else w for w in ws),bool(paragraph_first))
            hit=self._feature_cache.get(key)
            if hit is not None: rows[idx]=hit
            else:
                misses.append(idx);miss_words.append(ws);miss_keys.append(key)
        if misses:
            langs,supports_raw=self.s.batch_language_support(miss_words,max_order=5,slot_aware=True)
            for pos,ws,key,lang,raw_support in zip(misses,miss_words,miss_keys,langs,supports_raw):
                n=len(ws);lp=self.length_weight*self.s.length_logprior(n)
                opening=" ".join(w for w in ws[:3] if not self.s.is_slot(w))
                support=self.support_weight*float(raw_support)
                op=self.opening_weight*self.opening_score(ws)
                cscore,ccov,chits=self.grammar.slot_frame_score_tokens(ws,self.s.is_slot)
                pscore,pmeta=self.grammar.opening_position_score_tokens(ws,paragraph_first,self.s.is_slot)
                hit=(ws,n,float(lang),lp,opening,support,op,cscore,ccov,chits,pscore,pmeta)
                self._feature_cache[key]=hit;rows[pos]=hit
        return rows

    def choose(self,candidates,recent_openings=(),recent_templates=(),paragraph_first=False):
        best=None;ro=Counter(recent_openings);rt=Counter(recent_templates)
        features=self._static_features_many([x[0] for x in candidates],paragraph_first)
        for (text,meta),feat in zip(candidates,features):
            ws,n,lang,lp,opening,support,op,cscore,ccov,chits,pscore,pmeta=feat
            rep=self.repetition_weight*ro.get(opening,0)
            trep=self.template_repetition_weight*rt.get(meta.get("template"),0)
            target=meta.get("target_length")
            tpen=0.0 if target is None else self.target_weight*abs(n-target)/max(8.0,float(target))
            pbonus=self.proposal_weight*float(meta.get("proposal_confidence",0.0))
            score=(lang+lp+support+op-rep-trep-tpen+
                   self.construction_weight*cscore+self.position_weight*pscore+pbonus)
            meta2=dict(meta);meta2["construction_score"]=cscore;meta2["construction_coverage"]=ccov
            meta2["construction_hits"]=chits;meta2["position_evidence"]=pmeta;meta2["paragraph_first"]=bool(paragraph_first)
            meta2["proposal_bonus"]=pbonus;meta2["compute_backend"]="cuda"
            row=(score,self.rng.random(),text,meta2,lang,lp,support,rep,op,tpen,trep,cscore,pscore,pbonus)
            if best is None or row[:2]>best[:2]:best=row
        return best


def build_renderer_v9_gpu(root,seed=101,use_hot=False,proposal_weight=.06,device=0,memory_limit_mb=4608):
    root=Path(root)
    scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists():
        inducer=SafeWrapperInducer(root,scorer)
    else:
        inducer=StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector=LearnedSurfaceSelectorGPU(
        scorer,seed=seed,grammar=grammar,
        repetition_weight=RENDERER_V8_CONFIG['repetition_weight'],
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V8_CONFIG['template_repetition_weight'],
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],
        position_weight=RENDERER_V8_CONFIG['position_weight'],proposal_weight=proposal_weight)
    planner=EmpiricalStructurePlannerV2(
        scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],
        target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,inducer,RendererV9(selector,planner,proposer)

class RendererV9GPU(RendererV9):
    """V9 renderer that prewarms all static candidate features of a document in CUDA batches."""
    def render(self, facts):
        groups,targets=self.structure.bundle(facts)
        prepared=[]; induced_candidates=0; cur_focus=None
        # Pass 1: create/dedupe all candidates and determine paragraph position.
        for g,target in zip(groups,targets):
            focus=g[0][1]; paragraph_first=(cur_focus is None or focus!=cur_focus)
            cands=[]
            for text,meta in focus_bundle_candidates(g):
                meta=dict(meta);meta['target_length']=target;meta.setdefault('source','verified_v8')
                cands.append((text,meta))
            new=self.proposer.propose(g,target);induced_candidates+=len(new);cands.extend(new)
            ded=[];seen=set()
            for row in cands:
                if row[0] in seen:continue
                seen.add(row[0]);ded.append(row)
            prepared.append((g,target,focus,paragraph_first,ded))
            cur_focus=focus

        # Two large GPU batches are enough for all language/static features in the document.
        bypos={False:[],True:[]}
        for _,_,_,pf,cands in prepared:
            bypos[pf].extend(x[0] for x in cands)
        for pf,texts in bypos.items():
            if texts:self.sel._static_features_many(texts,pf)

        # Pass 2: sequential dynamic scoring preserves repetition memory exactly.
        sentences=[];represented=[];picks=[];recent=[];recent_t=[];paragraphs=[];cur_focus=None;cur=[]
        induced_selected=0
        for g,target,focus,paragraph_first,ded in prepared:
            pick=self.sel.choose(ded,recent,recent_t,paragraph_first=paragraph_first)
            text=pick[2]
            if pick[3].get('source')=='induced_wrapper':induced_selected+=1
            sentences.append(text);represented.extend(pick[3]['facts']);picks.append(pick)
            ws=self.sel.s.tokenize(text);opening=' '.join(w for w in ws[:3] if not self.sel.s.is_slot(w))
            recent=(recent+[opening])[-12:];recent_t=(recent_t+[pick[3]['template']])[-12:]
            if cur_focus is None:cur_focus=focus
            if focus!=cur_focus:
                paragraphs.append(' '.join(cur));cur=[];cur_focus=focus
            cur.append(text)
        if cur:paragraphs.append(' '.join(cur))
        return {'text':'\n\n'.join(paragraphs),'sentences':sentences,'paragraphs':paragraphs,
                'represented':represented,'picks':picks,'groups':groups,'targets':targets,
                'induced_candidates':induced_candidates,'induced_selected':induced_selected,
                'compute_backend':'cuda-batched'}


def build_renderer_v9_gpu_batched(root,seed=101,use_hot=False,proposal_weight=.06,device=0,memory_limit_mb=4608):
    root=Path(root)
    scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists():
        inducer=SafeWrapperInducer(root,scorer)
    else:
        inducer=StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector=LearnedSurfaceSelectorGPU(
        scorer,seed=seed,grammar=grammar,
        repetition_weight=RENDERER_V8_CONFIG['repetition_weight'],
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V8_CONFIG['template_repetition_weight'],
        construction_weight=RENDERER_V8_CONFIG['construction_weight'],
        position_weight=RENDERER_V8_CONFIG['position_weight'],proposal_weight=proposal_weight)
    planner=EmpiricalStructurePlannerV2(
        scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],
        target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,inducer,RendererV9GPU(selector,planner,proposer)


class GpuLocalOrderVerifier:
    """Corpus-calibrated local word-order verifier using the CUDA p2-p5 scorer.

    It contains no language-specific rules. A sentence is suspicious when an adjacent
    non-slot swap inside a local window improves the learned corpus score by more than
    the externally calibrated threshold.
    """
    def __init__(self, scorer, threshold=0.6576928743724331, window=9):
        self.s=scorer; self.threshold=float(threshold); self.window=int(window)

    def anomaly_score_tokens(self, words):
        ws=list(words); bases=[]; variants=[]; positions=[]
        for j in range(len(ws)-1):
            if self.s.is_slot(ws[j]) or self.s.is_slot(ws[j+1]) or ws[j]==ws[j+1]:
                continue
            lo=max(0,j-self.window//2); hi=min(len(ws),lo+self.window); lo=max(0,hi-self.window)
            base=ws[lo:hi]; k=j-lo
            if k<0 or k+1>=len(base): continue
            sw=list(base); sw[k],sw[k+1]=sw[k+1],sw[k]
            bases.append(base);variants.append(sw);positions.append(j)
        if not variants:return 0.0,None
        langs,_=self.s.batch_language_support(bases+variants,max_order=5,slot_aware=True)
        n=len(bases);gains=langs[n:]-langs[:n]
        idx=int(np.argmax(gains))
        return float(gains[idx]),positions[idx]

    def inspect_tokens(self, words):
        score,pos=self.anomaly_score_tokens(words)
        return [] if score<=self.threshold else [{'gain':score,'swap_index':pos}]

    def inspect(self,text):
        return self.inspect_tokens(self.s.tokenize(text))

class SemanticTraceVerifier:
    """Generic integrity verifier for the immutable semantic realization trace.

    The renderer stores the ordered protected-slot trace of the selected verified
    realization. This detects role swaps, value swaps, insertions and deletions without
    knowing the meaning of any particular relation/property token.
    """
    RX = re.compile(r'\b(?:e\d+|a\d+|v\d+|r\d+)\b', re.I)

    @classmethod
    def trace(cls,text):
        return tuple(x.lower() for x in cls.RX.findall(text))

    def inspect_render(self,out):
        bad=[]
        for i,(text,pick) in enumerate(zip(out['sentences'],out['picks'])):
            expected=pick[3].get('semantic_slot_trace')
            if expected is None:
                expected=self.trace(text)
                pick[3]['semantic_slot_trace']=expected
            if self.trace(text)!=tuple(expected):bad.append(i)
        return bad

    def inspect_sentence(self,text,pick):
        expected=pick[3].get('semantic_slot_trace')
        if expected is None:return False
        return self.trace(text)==tuple(expected)


def attach_semantic_traces(out):
    for text,pick in zip(out.get('sentences',()),out.get('picks',())):
        pick[3]['semantic_slot_trace']=SemanticTraceVerifier.trace(text)
    return out


class RendererV10GPU(RendererV9GPU):
    """GPU renderer with immutable semantic slot traces attached to every selection."""
    def render(self,facts):
        return attach_semantic_traces(super().render(facts))


def build_renderer_v10_gpu(root,seed=101,use_hot=False,proposal_weight=.06,device=0,memory_limit_mb=4608):
    scorer,grammar,inducer,r=build_renderer_v9_gpu_batched(
        root,seed=seed,use_hot=use_hot,proposal_weight=proposal_weight,
        device=device,memory_limit_mb=memory_limit_mb)
    return scorer,grammar,inducer,RendererV10GPU(r.sel,r.structure,r.proposer)


def build_renderer_v11_gpu(root,seed=101,use_hot=False,proposal_weight=.24,position_weight=.15,
                           construction_weight=None,device=0,memory_limit_mb=4608):
    root=Path(root)
    scorer=GpuBagacoSurfaceScorer(root,use_hot=use_hot,device=device,memory_limit_mb=memory_limit_mb)
    grammar=InducedConstructionGrammar(root)
    if (root/'model'/'quality'/'open.jsonl').exists() and (root/'model'/'full'/'open.jsonl').exists():
        inducer=SafeWrapperInducer(root,scorer)
    else:
        inducer=StoredWrapperInducer(root,scorer)
    proposer=InducedRealizationProposer(inducer)
    selector=LearnedSurfaceSelectorGPU(
        scorer,seed=seed,grammar=grammar,
        repetition_weight=RENDERER_V8_CONFIG['repetition_weight'],
        target_weight=RENDERER_V8_CONFIG['target_weight'],
        template_repetition_weight=RENDERER_V8_CONFIG['template_repetition_weight'],
        construction_weight=(RENDERER_V8_CONFIG['construction_weight'] if construction_weight is None else float(construction_weight)),
        position_weight=float(position_weight),proposal_weight=float(proposal_weight))
    planner=EmpiricalStructurePlannerV2(
        scorer,seed=seed,max_bundle=RENDERER_V8_CONFIG['max_bundle'],
        q_low=RENDERER_V8_CONFIG['q_low'],q_high=RENDERER_V8_CONFIG['q_high'],
        target_scale=RENDERER_V8_CONFIG['target_scale'])
    return scorer,grammar,inducer,RendererV10GPU(selector,planner,proposer)
