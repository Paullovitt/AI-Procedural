import json, time, os
from pathlib import Path
import numpy as np
import psutil
from numba import njit

ROOT=Path(__file__).resolve().parent
TRAIN=ROOT/'data'/'train.bin'
OUT=ROOT/'runs'/'procedural_direct_80m_32k'
OUT.mkdir(parents=True,exist_ok=True)
WINDOW=32768
CHUNK=64*1024*1024
BASE_STATES=256+65536+16777216
EXTRA_STATES=80_000_000-BASE_STATES
SLOTS=EXTRA_STATES//3
META_STATES=EXTRA_STATES-SLOTS*3
TABLE_CAP=1<<17
REBUILD_EVERY=32768
assert BASE_STATES + 3*SLOTS + META_STATES == 80_000_000
assert META_STATES == 2

@njit(cache=True)
def mix64(x):
    x=np.uint64(x+np.uint64(0x9E3779B97F4A7C15))
    x=np.uint64((x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9))
    x=np.uint64((x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB))
    return np.uint64(x^(x>>np.uint64(31)))

@njit(cache=True)
def sparse_add(key_lo,key_hi,count,key40):
    lo=np.uint32(key40 & np.uint64(0xffffffff))
    hi=np.uint32((key40>>np.uint64(32))+np.uint64(1)) # zero means empty
    idx=np.int64(mix64(key40)%np.uint64(SLOTS))
    probes=0
    while True:
        h=key_hi[idx]
        if h==0:
            key_lo[idx]=lo; key_hi[idx]=hi; count[idx]=1
            return 1,probes
        if h==hi and key_lo[idx]==lo:
            if count[idx] < np.uint32(0xffffffff): count[idx]+=1
            return 0,probes
        idx+=1; probes+=1
        if idx==SLOTS: idx=0
        if probes>=SLOTS: return -1,probes

@njit(cache=True)
def pair_find(ctx_keys,targets,counts,states,key,target):
    cap=len(ctx_keys)
    idx=np.int64(mix64(key^(np.uint64(target)*np.uint64(0x9E3779B97F4A7C15))) & np.uint64(cap-1))
    first_tomb=np.int64(-1)
    while True:
        st=states[idx]
        if st==0:
            return -(first_tomb+1) if first_tomb>=0 else -(idx+1)
        if st==1 and ctx_keys[idx]==key and targets[idx]==target: return idx
        if st==2 and first_tomb<0: first_tomb=idx
        idx=np.int64((idx+np.int64(1))&np.int64(cap-1))

@njit(cache=True)
def assoc_add(ctx_keys,targets,counts,states,key,target):
    s=pair_find(ctx_keys,targets,counts,states,key,target)
    if s>=0:
        if counts[s]<65535: counts[s]+=1
        return 0,1
    idx=-s-1;ctx_keys[idx]=key;targets[idx]=target;counts[idx]=1;states[idx]=1
    return 1,0

@njit(cache=True)
def assoc_remove(ctx_keys,targets,counts,states,key,target):
    s=pair_find(ctx_keys,targets,counts,states,key,target)
    if s<0:return 0
    if counts[s]>1:counts[s]-=1;return 0
    counts[s]=0;states[s]=2;return -1

@njit(cache=True)
def rebuild_assoc(ctx_keys,targets,counts,states,ring_keys,ring_targets,ring_count):
    states.fill(0);counts.fill(0);active=0
    for i in range(ring_count):
        d,_=assoc_add(ctx_keys,targets,counts,states,ring_keys[i],ring_targets[i]);active+=d
    return active

@njit(cache=True)
def scan_chunk(data,unigram,bigram,trigram,key_lo,key_hi,order4_count,
               ctx_keys,targets,acounts,states,ring_keys,ring_targets,
               roll8,seen,ring_pos,ring_count,active_pairs,pair_hits,removes,occupied,total_probes,max_probe):
    for j in range(len(data)):
        b=int(data[j]);unigram[b]+=1
        if seen>=1:
            p1=int(roll8 & np.uint64(0xff));bigram[(p1<<8)|b]+=1
        if seen>=2:
            p2=int((roll8>>np.uint64(8))&np.uint64(0xff));p1=int(roll8&np.uint64(0xff))
            idx=(p2<<16)|(p1<<8)|b
            if trigram[idx]<np.uint32(0xffffffff):trigram[idx]+=1
        if seen>=4:
            ctx4=np.uint64(roll8&np.uint64(0xffffffff));key40=(ctx4<<np.uint64(8))|np.uint64(b)
            d,pr=sparse_add(key_lo,key_hi,order4_count,key40)
            if d<0: return roll8,seen,ring_pos,ring_count,active_pairs,pair_hits,removes,occupied,total_probes,max_probe,-1
            occupied+=d;total_probes+=pr
            if pr>max_probe:max_probe=pr
        if seen>=8:
            key8=roll8
            if ring_count>=WINDOW:
                active_pairs+=assoc_remove(ctx_keys,targets,acounts,states,ring_keys[ring_pos],ring_targets[ring_pos])
            else:ring_count+=1
            d,hit=assoc_add(ctx_keys,targets,acounts,states,key8,np.uint8(b));active_pairs+=d;pair_hits+=hit
            ring_keys[ring_pos]=key8;ring_targets[ring_pos]=np.uint8(b);ring_pos+=1
            if ring_pos==WINDOW:ring_pos=0
            if ring_count>=WINDOW:
                removes+=1
                if removes>=REBUILD_EVERY:
                    active_pairs=rebuild_assoc(ctx_keys,targets,acounts,states,ring_keys,ring_targets,ring_count);removes=0
        roll8=np.uint64((roll8<<np.uint64(8))|np.uint64(b));seen+=1
    return roll8,seen,ring_pos,ring_count,active_pairs,pair_hits,removes,occupied,total_probes,max_probe,0

def main():
    arr=np.memmap(TRAIN,dtype=np.uint8,mode='r');N=len(arr)
    lim=int(os.environ.get('DIRECT_LIMIT','0'))
    if lim>0:N=min(N,lim)
    unigram=np.zeros(256,np.uint32);bigram=np.zeros(65536,np.uint32);trigram=np.zeros(1<<24,np.uint32)
    key_lo=np.zeros(SLOTS,np.uint32);key_hi=np.zeros(SLOTS,np.uint32);order4_count=np.zeros(SLOTS,np.uint32);meta=np.zeros(META_STATES,np.uint32)
    ctx_keys=np.zeros(TABLE_CAP,np.uint64);targets=np.zeros(TABLE_CAP,np.uint8);acounts=np.zeros(TABLE_CAP,np.uint16);states=np.zeros(TABLE_CAP,np.uint8)
    ring_keys=np.zeros(WINDOW,np.uint64);ring_targets=np.zeros(WINDOW,np.uint8)
    # JIT warmup on real signature, then reset.
    tcomp=time.perf_counter();wu=np.arange(64,dtype=np.uint8)
    scan_chunk(wu,unigram,bigram,trigram,key_lo,key_hi,order4_count,ctx_keys,targets,acounts,states,ring_keys,ring_targets,np.uint64(0),np.int64(0),np.int64(0),np.int64(0),np.int64(0),np.int64(0),np.int64(0),np.int64(0),np.int64(0),np.int64(0));compile_sec=time.perf_counter()-tcomp
    for x in (unigram,bigram,trigram,key_lo,key_hi,order4_count,ctx_keys,targets,acounts,states,ring_keys,ring_targets):x.fill(0)
    roll8=np.uint64(0);seen=ring_pos=ring_count=active_pairs=pair_hits=removes=occupied=total_probes=max_probe=0
    proc=psutil.Process();rss0=proc.memory_info().rss/1048576;t0=time.perf_counter()
    for start in range(0,N,CHUNK):
        end=min(N,start+CHUNK);data=np.asarray(arr[start:end],dtype=np.uint8)
        args=(np.uint64(roll8),np.int64(seen),np.int64(ring_pos),np.int64(ring_count),np.int64(active_pairs),np.int64(pair_hits),np.int64(removes),np.int64(occupied),np.int64(total_probes),np.int64(max_probe))
        roll8,seen,ring_pos,ring_count,active_pairs,pair_hits,removes,occupied,total_probes,max_probe,err=scan_chunk(data,unigram,bigram,trigram,key_lo,key_hi,order4_count,ctx_keys,targets,acounts,states,ring_keys,ring_targets,*args)
        if err: raise RuntimeError('Sparse order4 table full')
        sec=time.perf_counter()-t0
        print(json.dumps({'bytes':end,'total':N,'pct':100*end/N,'sec':sec,'MiB_s':end/1048576/sec,'order4_entries':int(occupied),'load':float(occupied/SLOTS),'avg_probe':float(total_probes/max(1,seen-4)),'max_probe':int(max_probe),'assoc_hits':int(pair_hits),'rss_mb':proc.memory_info().rss/1048576}),flush=True)
    pass_sec=time.perf_counter()-t0
    meta[0]=np.uint32(occupied);meta[1]=np.uint32(min(total_probes,0xffffffff))
    np.save(OUT/'unigram.npy',unigram);np.save(OUT/'bigram.npy',bigram.reshape(256,256));np.save(OUT/'trigram.npy',trigram.reshape(65536,256))
    np.save(OUT/'order4_key_lo.npy',key_lo);np.save(OUT/'order4_key_hi.npy',key_hi);np.save(OUT/'order4_count.npy',order4_count);np.save(OUT/'meta.npy',meta)
    summary={'experiment':'Direct-80M-32k exact-sparse','train':str(TRAIN),'bytes':N,'persistent_scalar_states':80_000_000,'persistent_dtype':'uint32','persistent_storage_bytes_raw':320_000_000,'state_breakdown':{'unigram':256,'bigram':65536,'trigram':16777216,'order4_key_lo':SLOTS,'order4_key_hi':SLOTS,'order4_count':SLOTS,'meta':META_STATES},'order4_slots':SLOTS,'order4_entries':int(occupied),'order4_load':float(occupied/SLOTS),'order4_avg_probe':float(total_probes/max(1,N-4)),'order4_max_probe':int(max_probe),'order4_semantics':'exact verified sparse table: previous 4 bytes + target byte -> count','runtime_context_window':WINDOW,'runtime_copy_key_bytes':8,'active_associative_pairs_final':int(active_pairs),'assoc_pair_hits':int(pair_hits),'gradient_steps':0,'optimizer':None,'full_pass_seconds':pass_sec,'full_pass_minutes':pass_sec/60,'MiB_s':N/1048576/pass_sec,'jit_compile_seconds_excluded':compile_sec,'rss_mb_before':rss0,'rss_mb_after':proc.memory_info().rss/1048576,'note':'Exactly 80,000,000 persistent uint32 scalar states (~320 MB raw), not neural parameters. 32k associative memory is runtime context and is excluded from the 80M persistent-state count.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('SUMMARY '+json.dumps(summary),flush=True)
if __name__=='__main__':main()
