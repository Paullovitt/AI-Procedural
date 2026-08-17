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

def sinusoidal_positions(t,d,device,dtype):
    pos=torch.arange(t,device=device,dtype=torch.float32).unsqueeze(1)
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
        gh=max(32,cfg.d_model//4); self.gate=nn.Sequential(nn.Linear(cfg.d_model,gh),nn.SiLU(),nn.Linear(gh,4*self.r))
        self.seed=1009.0+211.0*layer_id
    def forward(self,x):
        b,t,c=x.shape
        gq,gk,gv,go=self.gate(x).chunk(4,-1)
        gq=1+.5*torch.tanh(gq); gk=1+.5*torch.tanh(gk); gv=1+.5*torch.tanh(gv); go=1+.5*torch.tanh(go)
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
    def forward(self,x):
        g1,g2=self.gate(x).chunk(2,-1); g1=1+.5*torch.tanh(g1); g2=1+.5*torch.tanh(g2)
        A=procedural_matrix(self.d,self.r,self.seed+1,x.device,x.dtype,self.d); y=x@A; del A; y=y*g1
        B=procedural_matrix(self.r,self.h,self.seed+2,x.device,x.dtype,self.r); y=y@B; del B; y=F.gelu(y)
        C=procedural_matrix(self.h,self.r,self.seed+3,x.device,x.dtype,self.h); y=y@C; del C; y=y*g2
        D=procedural_matrix(self.r,self.d,self.seed+4,x.device,x.dtype,self.r); y=y@D; del D
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
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>self.cfg.seq_len: raise ValueError(f'sequence {t} > configured seq_len {self.cfg.seq_len}')
        x=self.drop(self.tok(idx)+sinusoidal_positions(t,self.cfg.d_model,idx.device,self.tok.weight.dtype))
        for blk in self.blocks: x=blk(x)
        logits=self.lm_head(self.ln_f(x))
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
    def parameter_report(self):
        return {'parameters':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}
