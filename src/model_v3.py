from dataclasses import dataclass
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelConfig:
    vocab_size:int=256; d_model:int=256; n_layers:int=6; n_heads:int=8; seq_len:int=256
    ffn_mult:int=4; dropout:float=0.1; procedural_rank:int=32
    model_type:str='procedural'; activation_checkpoint:bool=False; fused_qkv:bool=True
    latent_attention:bool=True; cache_procedural:bool=True; ffn_active_dims:int=0
    v3_hybrid:bool=True; v3_beta:float=0.5; v3_eps:float=1e-6

def sinusoidal_positions(t,d,device,dtype,offset=0):
    pos=torch.arange(offset,offset+t,device=device,dtype=torch.float32).unsqueeze(1)
    inv=torch.exp(torch.arange(0,d,2,device=device,dtype=torch.float32)*(-math.log(10000.0)/d))
    out=torch.zeros((t,d),device=device,dtype=torch.float32)
    out[:,0::2]=torch.sin(pos*inv); out[:,1::2]=torch.cos(pos*inv)
    return out.to(dtype=dtype)

def procedural_matrix(rows,cols,seed,device,dtype,normalize_by):
    i=torch.arange(rows,device=device,dtype=torch.float32).unsqueeze(1)+1.0
    j=torch.arange(cols,device=device,dtype=torch.float32).unsqueeze(0)+1.0
    x=i*(0.017+seed*0.00011)+j*(0.031+seed*0.00007)
    m=torch.sin(x*(1.0+(i%7)*0.013))+0.5*torch.cos(x*1.731+seed)
    return (m/math.sqrt(float(normalize_by))).to(dtype=dtype)

def proc_project(x,gate,rank,seed,out_dim=None):
    d=x.size(-1); out_dim=d if out_dim is None else out_dim
    A=procedural_matrix(d,rank,seed+1,x.device,x.dtype,d); y=x@A; del A
    y=y*gate
    B=procedural_matrix(rank,out_dim,seed+2,x.device,x.dtype,rank); y=y@B; del B
    return y

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); assert cfg.d_model%cfg.n_heads==0
        self.h=cfg.n_heads; self.hd=cfg.d_model//cfg.n_heads; self.drop=cfg.dropout
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model,bias=False); self.out=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,-1)
        q=q.view(b,t,self.h,self.hd).transpose(1,2); k=k.view(b,t,self.h,self.hd).transpose(1,2); v=v.view(b,t,self.h,self.hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.drop if self.training else 0.0,is_causal=True)
        return self.out(y.transpose(1,2).contiguous().view(b,t,c))

class ProceduralSelfAttention(nn.Module):
    def __init__(self,cfg,layer_id):
        super().__init__(); assert cfg.d_model%cfg.n_heads==0
        self.d=cfg.d_model; self.r=cfg.procedural_rank; self.h=cfg.n_heads; self.hd=cfg.d_model//cfg.n_heads; self.drop=cfg.dropout; self.fused=cfg.fused_qkv
        self.latent=getattr(cfg,'latent_attention',True)
        self.v3_hybrid=bool(getattr(cfg,'v3_hybrid',False)); self.v3_beta=float(getattr(cfg,'v3_beta',0.5)); self.v3_eps=float(getattr(cfg,'v3_eps',1e-6))
        gh=max(32,cfg.d_model//4); self.gate=nn.Sequential(nn.Linear(cfg.d_model,gh),nn.SiLU(),nn.Linear(gh,4*self.r))
        self.seed=1009.0+211.0*layer_id
        # Caches/estado somente de runtime; nao entram no state_dict/checkpoint.
        self._latent_cache={}; self._v3_oldS=None; self._v3_oldz=None; self._v3_recent_k=None; self._v3_recent_v=None

    def reset_context(self):
        self._v3_oldS=self._v3_oldz=self._v3_recent_k=self._v3_recent_v=None

    def set_context_beta(self,beta):
        self.v3_beta=float(beta)

    def snapshot_context(self,to_cpu=False):
        vals=(self._v3_oldS,self._v3_oldz,self._v3_recent_k,self._v3_recent_v)
        out=[]
        for x in vals:
            if x is None: out.append(None)
            else:
                y=x.detach().clone()
                out.append(y.cpu() if to_cpu else y)
        return tuple(out)

    def restore_context(self,state,device=None):
        vals=[]
        for x in state:
            if x is None: vals.append(None)
            else: vals.append(x.to(device=device) if device is not None else x)
        self._v3_oldS,self._v3_oldz,self._v3_recent_k,self._v3_recent_v=vals

    def _latent_matrices(self,x):
        key=(str(x.device),x.dtype)
        cached=self._latent_cache.get(key)
        if cached is not None: return cached
        A=procedural_matrix(self.d,self.r,self.seed+1,x.device,x.dtype,self.d)
        Bq=procedural_matrix(self.r,self.d,self.seed+2,x.device,x.dtype,self.r)
        Bk=procedural_matrix(self.r,self.d,self.seed+12,x.device,x.dtype,self.r)
        Bv=procedural_matrix(self.r,self.d,self.seed+22,x.device,x.dtype,self.r)
        Aout=procedural_matrix(self.d,self.r,self.seed+31,x.device,x.dtype,self.d)
        Bout=procedural_matrix(self.r,self.d,self.seed+32,x.device,x.dtype,self.r)
        metrics=[]; collapses=[]
        for head in range(self.h):
            lo=head*self.hd; hi=lo+self.hd
            metrics.append(Bq[:,lo:hi]@Bk[:,lo:hi].transpose(0,1))
            collapses.append(Bv[:,lo:hi]@Aout[lo:hi,:])
        cached={'A':A,'metric':torch.stack(metrics,0),'collapse':torch.stack(collapses,0),'Bout':Bout}
        self._latent_cache[key]=cached
        return cached

    def _forward_latent(self,x,gq,gk,gv,go):
        mats=self._latent_matrices(x)
        z=x@mats['A']
        q=z*gq; k=z*gk; v=z*gv
        # q_metric @ k.T == Q_h @ K_h.T do caminho original.
        q_metric=torch.einsum('btr,hrs->bhts',q,mats['metric'])
        # SDPA escalaria por sqrt(rank). Ajuste q para preservar exatamente
        # a escala original 1/sqrt(head_dim), inclusive em Torch sem arg scale=.
        q_metric=q_metric*math.sqrt(float(self.r)/float(self.hd))
        k_heads=k.unsqueeze(1).expand(-1,self.h,-1,-1)
        v_heads=v.unsqueeze(1).expand(-1,self.h,-1,-1)
        weighted_v=F.scaled_dot_product_attention(q_metric,k_heads,v_heads,dropout_p=self.drop if self.training else 0.0,is_causal=True)
        compact=torch.einsum('bhtr,hrs->bts',weighted_v,mats['collapse'])
        compact=compact*go
        return compact@mats['Bout']

    def _forward_hybrid_v3(self,x,gq,gk,gv,go):
        mats=self._latent_matrices(x)
        z=x@mats['A']; q=z*gq; k=z*gk; v=z*gv
        qstd=torch.einsum('btr,hrs->bhts',q,mats['metric'])*math.sqrt(float(self.r)/float(self.hd))
        ks=[k]; vs=[v]; mem=0
        if self._v3_recent_k is not None:
            ks.insert(0,self._v3_recent_k); vs.insert(0,self._v3_recent_v); mem=self._v3_recent_k.size(1)
        kk=torch.cat(ks,1).unsqueeze(1).expand(-1,self.h,-1,-1)
        vv=torch.cat(vs,1).unsqueeze(1).expand(-1,self.h,-1,-1)
        t=x.size(1)
        if mem:
            cur=torch.ones((t,t),device=x.device,dtype=torch.bool).tril()
            mask=torch.cat((torch.ones((t,mem),device=x.device,dtype=torch.bool),cur),1)[None,None]
            local=F.scaled_dot_product_attention(qstd,kk,vv,attn_mask=mask,dropout_p=self.drop if self.training else 0.0)
        else:
            local=F.scaled_dot_product_attention(qstd,kk,vv,is_causal=True,dropout_p=self.drop if self.training else 0.0)
        beta=self.v3_beta
        if self._v3_oldS is not None and beta>0.0:
            with torch.autocast(device_type=x.device.type,enabled=False):
                qlin=F.elu(qstd.float()/math.sqrt(float(self.r)))+1.0
                num=torch.einsum('bhtr,bhrs->bhts',qlin,self._v3_oldS)
                den=torch.einsum('bhtr,bhr->bht',qlin,self._v3_oldz).unsqueeze(-1).clamp_min(self.v3_eps)
                glob=num/den
            weighted=(1.0-beta)*local+beta*glob.to(local.dtype)
        else:
            weighted=local
        compact=torch.einsum('bhtr,hrs->bts',weighted,mats['collapse']); out=(compact*go)@mats['Bout']
        # Move o bloco anterior para o estado linear global. O estado e detached:
        # contexto longo no forward, BPTT truncado no limite entre chunks.
        if self._v3_recent_k is not None:
            with torch.no_grad(),torch.autocast(device_type=x.device.type,enabled=False):
                rk=(F.elu(self._v3_recent_k.float())+1.0).unsqueeze(1).expand(-1,self.h,-1,-1)
                rv=self._v3_recent_v.float().unsqueeze(1).expand(-1,self.h,-1,-1)
                addS=(rk.unsqueeze(-1)*rv.unsqueeze(-2)).sum(2); addz=rk.sum(2)
                self._v3_oldS=addS if self._v3_oldS is None else self._v3_oldS+addS
                self._v3_oldz=addz if self._v3_oldz is None else self._v3_oldz+addz
        self._v3_recent_k=k.detach(); self._v3_recent_v=v.detach()
        return out

    def forward(self,x):
        b,t,c=x.shape
        gq,gk,gv,go=self.gate(x).chunk(4,-1)
        gq=1+.5*torch.tanh(gq); gk=1+.5*torch.tanh(gk); gv=1+.5*torch.tanh(gv); go=1+.5*torch.tanh(go)
        if self.latent and self.fused:
            if self.v3_hybrid:
                return self._forward_hybrid_v3(x,gq,gk,gv,go)
            return self._forward_latent(x,gq,gk,gv,go)
        if self.fused:
            A=procedural_matrix(self.d,self.r,self.seed+1,x.device,x.dtype,self.d); z=x@A; del A
            def branch(g,off):
                B=procedural_matrix(self.r,self.d,self.seed+off,x.device,x.dtype,self.r); y=(z*g)@B; del B; return y
            q=branch(gq,2); k=branch(gk,12); v=branch(gv,22)
        else:
            q=proc_project(x,gq,self.r,self.seed); k=proc_project(x,gk,self.r,self.seed+10); v=proc_project(x,gv,self.r,self.seed+20)
        q=q.view(b,t,self.h,self.hd).transpose(1,2); k=k.view(b,t,self.h,self.hd).transpose(1,2); v=v.view(b,t,self.h,self.hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.drop if self.training else 0.0,is_causal=True)
        y=y.transpose(1,2).contiguous().view(b,t,c)
        return proc_project(y,go,self.r,self.seed+30)

class BaselineFFN(nn.Module):
    def __init__(self,cfg):
        super().__init__(); h=cfg.d_model*cfg.ffn_mult
        self.net=nn.Sequential(nn.Linear(cfg.d_model,h,bias=False),nn.GELU(),nn.Linear(h,cfg.d_model,bias=False),nn.Dropout(cfg.dropout))
    def forward(self,x): return self.net(x)

class ProceduralFFN(nn.Module):
    def __init__(self,cfg,layer_id):
        super().__init__(); self.d=cfg.d_model; self.h=cfg.d_model*cfg.ffn_mult; self.r=cfg.procedural_rank; self.drop=nn.Dropout(cfg.dropout)
        gh=max(32,cfg.d_model//4); self.gate=nn.Sequential(nn.Linear(cfg.d_model,gh),nn.SiLU(),nn.Linear(gh,2*self.r)); self.seed=17.0+97.0*layer_id
        self.cache_procedural=getattr(cfg,'cache_procedural',True); self.active_dims=int(getattr(cfg,'ffn_active_dims',0) or 0); self._matrix_cache={}; self._subset_cache={}
    def _matrices(self,x):
        if not self.cache_procedural:
            return (
                procedural_matrix(self.d,self.r,self.seed+1,x.device,x.dtype,self.d),
                procedural_matrix(self.r,self.h,self.seed+2,x.device,x.dtype,self.r),
                procedural_matrix(self.h,self.r,self.seed+3,x.device,x.dtype,self.h),
                procedural_matrix(self.r,self.d,self.seed+4,x.device,x.dtype,self.r),
            )
        key=(str(x.device),x.dtype)
        mats=self._matrix_cache.get(key)
        if mats is None:
            mats=(
                procedural_matrix(self.d,self.r,self.seed+1,x.device,x.dtype,self.d),
                procedural_matrix(self.r,self.h,self.seed+2,x.device,x.dtype,self.r),
                procedural_matrix(self.h,self.r,self.seed+3,x.device,x.dtype,self.h),
                procedural_matrix(self.r,self.d,self.seed+4,x.device,x.dtype,self.r),
            )
            self._matrix_cache[key]=mats
        return mats
    def _active_bc(self,B,C):
        m=self.active_dims
        if m<=0 or m>=self.h: return B,C,1.0
        key=(str(B.device),B.dtype,m)
        pair=self._subset_cache.get(key)
        if pair is None:
            # Amostragem quase-uniforme da base procedural. Em testes, esta
            # preservou muito melhor a funcao que subconjuntos pseudoaleatorios.
            idx=torch.floor((torch.arange(m,device=B.device,dtype=torch.float32)+0.5)*(self.h/float(m))).long().clamp_max(self.h-1)
            pair=(B.index_select(1,idx),C.index_select(0,idx))
            self._subset_cache[key]=pair
        return pair[0],pair[1],self.h/float(m)
    def forward(self,x):
        g1,g2=self.gate(x).chunk(2,-1); g1=1+.5*torch.tanh(g1); g2=1+.5*torch.tanh(g2)
        A,B,C,D=self._matrices(x); B,C,scale=self._active_bc(B,C)
        y=x@A; y=y*g1; y=y@B; y=F.gelu(y); y=(y@C)*scale; y=y*g2; y=y@D
        return self.drop(y)

class Block(nn.Module):
    def __init__(self,cfg,layer_id):
        super().__init__(); self.ln1=nn.LayerNorm(cfg.d_model); self.ln2=nn.LayerNorm(cfg.d_model); self.ckpt=cfg.activation_checkpoint
        if cfg.model_type=='baseline':
            self.attn=CausalSelfAttention(cfg); self.ffn=BaselineFFN(cfg)
        elif cfg.model_type=='procedural':
            self.attn=ProceduralSelfAttention(cfg,layer_id); self.ffn=ProceduralFFN(cfg,layer_id)
        else:
            raise ValueError(f'unknown model_type: {cfg.model_type}')
    def forward(self,x):
        if self.ckpt and self.training and torch.is_grad_enabled():
            x=x+checkpoint(self.attn,self.ln1(x),use_reentrant=False)
            x=x+checkpoint(self.ffn,self.ln2(x),use_reentrant=False)
        else:
            x=x+self.attn(self.ln1(x)); x=x+self.ffn(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.tok=nn.Embedding(cfg.vocab_size,cfg.d_model); self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([Block(cfg,i) for i in range(cfg.n_layers)]); self.ln_f=nn.LayerNorm(cfg.d_model); self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False); self.lm_head.weight=self.tok.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,.02)
    def reset_context(self):
        for blk in self.blocks:
            if hasattr(blk.attn,'reset_context'): blk.attn.reset_context()

    def set_context_beta(self,beta):
        for blk in self.blocks:
            if hasattr(blk.attn,'set_context_beta'): blk.attn.set_context_beta(beta)

    def snapshot_context(self,to_cpu=False):
        return [blk.attn.snapshot_context(to_cpu=to_cpu) if hasattr(blk.attn,'snapshot_context') else None for blk in self.blocks]

    def restore_context(self,state):
        device=self.tok.weight.device
        for blk,s in zip(self.blocks,state):
            if s is not None and hasattr(blk.attn,'restore_context'): blk.attn.restore_context(s,device=device)

    def forward(self,idx,targets=None,position_offset=0):
        b,t=idx.shape
        if t>self.cfg.seq_len: raise ValueError(f'sequence {t} > configured seq_len {self.cfg.seq_len}')
        x=self.drop(self.tok(idx)+sinusoidal_positions(t,self.cfg.d_model,idx.device,self.tok.weight.dtype,offset=position_offset))
        for blk in self.blocks: x=blk(x)
        logits=self.lm_head(self.ln_f(x))
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
    def parameter_report(self):
        return {'parameters':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}
