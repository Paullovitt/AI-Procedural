from collections import deque
from pathlib import Path
import numpy as np
from numba import njit

SLOTS=21_052_330

@njit(cache=True)
def _mix64(x):
    x=np.uint64(x+np.uint64(0x9E3779B97F4A7C15))
    x=np.uint64((x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9))
    x=np.uint64((x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB))
    return np.uint64(x^(x>>np.uint64(31)))

@njit(cache=True)
def _lookup_order4_256(key_lo,key_hi,count,ctx4):
    out=np.zeros(256,np.float64)
    base=np.uint64(ctx4)<<np.uint64(8)
    for b in range(256):
        key40=base|np.uint64(b)
        lo=np.uint32(key40&np.uint64(0xffffffff));hi=np.uint32((key40>>np.uint64(32))+np.uint64(1))
        idx=np.int64(_mix64(key40)%np.uint64(SLOTS))
        while True:
            h=key_hi[idx]
            if h==0: break
            if h==hi and key_lo[idx]==lo:
                out[b]=count[idx];break
            idx+=1
            if idx==SLOTS:idx=0
    return out

class Direct80M32K:
    def __init__(self,run_dir, kappa1=100.,kappa2=300.,kappa4=100.,copy_lambda=.75,copy_key=8,copy_window=32768,eps=1e-5):
        d=Path(run_dir)
        self.unigram=np.load(d/'unigram.npy',mmap_mode='r')
        self.bigram=np.load(d/'bigram.npy',mmap_mode='r')
        self.trigram=np.load(d/'trigram.npy',mmap_mode='r')
        self.key_lo=np.load(d/'order4_key_lo.npy',mmap_mode='r')
        self.key_hi=np.load(d/'order4_key_hi.npy',mmap_mode='r')
        self.order4_count=np.load(d/'order4_count.npy',mmap_mode='r')
        self.kappa1=float(kappa1);self.kappa2=float(kappa2);self.kappa4=float(kappa4)
        self.copy_lambda=float(copy_lambda);self.copy_key=int(copy_key);self.copy_window=int(copy_window);self.eps=float(eps)
        u=np.asarray(self.unigram,dtype=np.float64);self.p0=((u+.5)/(u.sum()+.5*256)).astype(np.float32)
        self.big_totals=np.asarray(self.bigram,dtype=np.uint64).sum(axis=1)
        self.tri_totals=np.asarray(self.trigram,dtype=np.uint64).sum(axis=1)
        self.reset_context()
    def reset_context(self):
        self.history=deque(maxlen=max(self.copy_key,4));self.copy={};self.copy_queue=deque();self.position=0
    def _global(self):
        p=self.p0.astype(np.float64,copy=True)
        if len(self.history)>=1:
            a=int(self.history[-1]);tot=float(self.big_totals[a])
            if tot>0:
                ml=np.asarray(self.bigram[a],dtype=np.float64)/tot;lam=tot/(tot+self.kappa1);p=lam*ml+(1-lam)*p
        if len(self.history)>=2:
            a,b=int(self.history[-2]),int(self.history[-1]);ctx=(a<<8)|b;tot=float(self.tri_totals[ctx])
            if tot>0:
                ml=np.asarray(self.trigram[ctx],dtype=np.float64)/tot;lam=tot/(tot+self.kappa2);p=lam*ml+(1-lam)*p
        if len(self.history)>=4:
            h=list(self.history);ctx4=(int(h[-4])<<24)|(int(h[-3])<<16)|(int(h[-2])<<8)|int(h[-1])
            c=_lookup_order4_256(self.key_lo,self.key_hi,self.order4_count,np.uint64(ctx4));tot=float(c.sum())
            if tot>0:
                ml=c/tot;lam=tot/(tot+self.kappa4);p=lam*ml+(1-lam)*p
        return p
    def _observe(self,byte_value):
        b=int(byte_value)
        if len(self.history)>=self.copy_key:
            key=bytes(self.history);row=self.copy.get(key)
            if row is None:self.copy[key]={b:1}
            else:row[b]=row.get(b,0)+1
            self.copy_queue.append((self.position,key,b))
        cutoff=self.position-self.copy_window
        while self.copy_queue and self.copy_queue[0][0]<=cutoff:
            _,key,t=self.copy_queue.popleft();row=self.copy[key];row[t]-=1
            if row[t]<=0:del row[t]
            if not row:del self.copy[key]
        self.history.append(b);self.position+=1
    def predict_after_observing(self,byte_value):
        self._observe(byte_value);p=self._global()
        if self.copy_lambda>0 and len(self.history)>=self.copy_key:
            key=bytes(self.history);row=self.copy.get(key)
            if row:
                total=float(sum(row.values()));den=total+self.eps*256.;cp=np.full(256,self.eps/den,np.float64)
                for b,c in row.items():cp[b]=(float(c)+self.eps)/den
                p=(1-self.copy_lambda)*p+self.copy_lambda*cp
        return p.astype(np.float32,copy=False)
    def predict_chunk(self,input_bytes):
        vals=np.asarray(input_bytes,dtype=np.uint8).reshape(-1);out=np.empty((len(vals),256),np.float32)
        for i,b in enumerate(vals):out[i]=self.predict_after_observing(int(b))
        return out
