import argparse, gc, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from model_v3 import ModelConfig, TinyTransformer

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'runs'/'procedural_v3_hybrid32k_512_5000'
OUT=BASE
OUT.mkdir(parents=True,exist_ok=True)

ap=argparse.ArgumentParser()
ap.add_argument('--resume',type=str,default='')
ap.add_argument('--end',type=int,default=5000)
a=ap.parse_args()

SEED=1337
CHUNK=512
CONTEXT_BLOCKS=63
EFFECTIVE_CONTEXT=CHUNK*CONTEXT_BLOCKS  # 32256 ~= 32k
STEPS=5000
LR=0.0012
MIN_LR=0.00012
WARMUP=250
EVAL_EVERY=500
DEV='cuda'

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
rng=np.random.RandomState(SEED)
train=np.memmap(ROOT/'data'/'train.bin',dtype=np.uint8,mode='r')
val=np.memmap(ROOT/'data'/'val.bin',dtype=np.uint8,mode='r')
step0=torch.load(BASE/'step0.pt',map_location='cpu',weights_only=False)
base_cfg=dict(step0['config']['model_config'])
base_cfg.update({'seq_len':CHUNK,'latent_attention':True,'cache_procedural':True,'activation_checkpoint':False,'ffn_active_dims':0,'v3_hybrid':True,'v3_beta':0.05,'v3_eps':1e-6})
cfg=ModelConfig(**base_cfg)
model=TinyTransformer(cfg).to(DEV); model.load_state_dict(step0['model'],strict=True)
try: opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.95),weight_decay=.1,fused=True)
except Exception: opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.95),weight_decay=.1)
scaler=torch.amp.GradScaler('cuda',enabled=True)

def width_for(step):
    # Conservative schedule: full FFN for the last half to recover final quality.
    if step<=500:return 512
    if step<=1500:return 1024
    if step<=2500:return 1536
    return 0

def beta_for(step):
    # Curriculum for long-memory reliance.
    if step<=500:return 0.05
    if step<=1500:return 0.15
    if step<=2500:return 0.30
    return 0.50

def lr_for(step):
    if step<=WARMUP:return LR*step/WARMUP
    p=(step-WARMUP)/(STEPS-WARMUP)
    return MIN_LR+0.5*(LR-MIN_LR)*(1+math.cos(math.pi*p))

def set_width(w):
    for b in model.blocks:
        b.ffn.active_dims=int(w); b.ffn._subset_cache.clear()

def batch(arr,off,n=CHUNK):
    z=np.asarray(arr[off:off+n+1],dtype=np.int64).copy()
    return torch.from_numpy(z[:-1])[None].to(DEV),torch.from_numpy(z[1:])[None].to(DEV)

def eval32k(beta):
    snap=model.snapshot_context(to_cpu=False); was=model.training
    oldw=model.blocks[0].ffn.active_dims
    model.eval(); set_width(0); model.set_context_beta(beta); model.reset_context()
    loss_sum=0.; correct=0; count=0
    start=500_000
    torch.cuda.synchronize(); t=time.perf_counter()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for j in range(CONTEXT_BLOCKS):
            x,y=batch(val,start+j*CHUNK,CHUNK)
            lg,ls=model(x,y,position_offset=j*CHUNK)
            loss_sum+=float(ls)*CHUNK; correct+=(lg.argmax(-1)==y).sum().item(); count+=CHUNK
    torch.cuda.synchronize(); sec=time.perf_counter()-t
    model.reset_context(); model.restore_context(snap); set_width(oldw); model.train(was)
    vl=loss_sum/count
    return {'loss':vl,'ppl':math.exp(vl),'accuracy':correct/count,'tokens':count,'sec':sec,'tokens_s':count/sec}

def eval512(batches=12):
    snap=model.snapshot_context(to_cpu=False); was=model.training
    oldw=model.blocks[0].ffn.active_dims; oldbeta=model.blocks[0].attn.v3_beta
    model.eval(); set_width(0); model.set_context_beta(0.0)
    loss_sum=0.;correct=0;count=0
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for j in range(batches):
            model.reset_context(); x,y=batch(val,100_000+j*4096,CHUNK);lg,ls=model(x,y,position_offset=0)
            loss_sum+=float(ls)*CHUNK;correct+=(lg.argmax(-1)==y).sum().item();count+=CHUNK
    model.reset_context();model.restore_context(snap);model.set_context_beta(oldbeta);set_width(oldw);model.train(was)
    vl=loss_sum/count
    return {'loss':vl,'ppl':math.exp(vl),'accuracy':correct/count,'tokens':count}

def payload(step,best,train_sec,wall_sec,window_start):
    return {
        'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
        'step':step,'best_val32k_loss':best,'train_sec':train_sec,'wall_sec':wall_sec,
        'window_start':window_start,'context':model.snapshot_context(to_cpu=True),
        'rng_numpy':rng.get_state(),'rng_python':random.getstate(),'rng_torch':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all(),'config':{'model_config':base_cfg,'trainer':{
            'steps':STEPS,'logical_epochs':10,'chunk':CHUNK,'context_blocks':CONTEXT_BLOCKS,
            'effective_context':EFFECTIVE_CONTEXT,'lr':LR,'min_lr':MIN_LR,'warmup':WARMUP,
            'ffn_schedule':'512@1-500,1024@501-1500,1536@1501-2500,full@2501-5000',
            'beta_schedule':'0.05@1-500,0.15@501-1500,0.30@1501-2500,0.50@2501-5000'}}}

start=0;best=1e9;train_sec=0.;wall_prior=0.;window_start=None
if a.resume:
    c=torch.load(a.resume,map_location='cpu',weights_only=False)
    model.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);scaler.load_state_dict(c['scaler'])
    start=int(c['step']);best=float(c.get('best_val32k_loss',best));train_sec=float(c.get('train_sec',0));wall_prior=float(c.get('wall_sec',0));window_start=c.get('window_start')
    rng.set_state(c['rng_numpy']);random.setstate(c['rng_python']);torch.set_rng_state(c['rng_torch']);torch.cuda.set_rng_state_all(c['rng_cuda']);model.restore_context(c['context'])

# Baseline metrics are evaluated only for a fresh run and persisted separately.
if start==0:
    model.set_context_beta(beta_for(1)); model.reset_context(); set_width(0)
    initial32=eval32k(beta_for(1)); initial512=eval512()
    (OUT/'initial_metrics.json').write_text(json.dumps({'v3_32k':initial32,'exact512':initial512},indent=2),encoding='utf-8')
    print('INITIAL',json.dumps({'v3_32k':initial32,'exact512':initial512}),flush=True)
    model.reset_context()

model.train(); wall0=time.perf_counter(); rows=[]
torch.cuda.reset_peak_memory_stats()
for step in range(start+1,a.end+1):
    ci=(step-1)%CONTEXT_BLOCKS
    if ci==0:
        model.reset_context()
        window_start=int(rng.randint(0,len(train)-EFFECTIVE_CONTEXT-2))
    beta=beta_for(step);width=width_for(step);lr=lr_for(step)
    model.set_context_beta(beta);set_width(width)
    for g in opt.param_groups:g['lr']=lr
    off=window_start+ci*CHUNK;x,y=batch(train,off,CHUNK)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();ts=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.float16):_,loss=model(x,y,position_offset=ci*CHUNK)
    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update()
    torch.cuda.synchronize();train_sec+=time.perf_counter()-ts
    if step%EVAL_EVERY==0 or step==a.end:
        wall=wall_prior+(time.perf_counter()-wall0)
        ev32=eval32k(beta);ev512=eval512();peak=torch.cuda.max_memory_allocated()/2**20
        row={'step':step,'logical_epoch':step//500,'width':width,'beta':beta,'train_loss':float(loss.detach()),'lr':lr,'tokens_seen':step*CHUNK,'train_sec':train_sec,'wall_sec':wall,'train_tokens_s':step*CHUNK/max(train_sec,1e-9),'peak_vram_mb':peak,'val32k':ev32,'exact512':ev512}
        rows.append(row);print('PROGRESS',json.dumps(row),flush=True)
        p=payload(step,min(best,ev32['loss']),train_sec,wall,window_start);torch.save(p,OUT/'last.pt')
        if ev32['loss']<best:
            best=ev32['loss'];torch.save(p,OUT/'best.pt')
        # Append-safe progress log independent of resume chunks.
        with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')

wall_total=wall_prior+(time.perf_counter()-wall0)
final_beta=beta_for(a.end);final32=eval32k(final_beta);final512=eval512();peak=torch.cuda.max_memory_allocated()/2**20
summary={'experiment':'procedural_v3_hybrid32k_512_5000','steps':a.end,'logical_epochs':a.end//500,'tokens_seen':a.end*CHUNK,'chunk_tokens':CHUNK,'effective_context_tokens':EFFECTIVE_CONTEXT,'context_blocks':CONTEXT_BLOCKS,'final_beta':final_beta,'final_width':width_for(a.end),'train_sec':train_sec,'wall_sec':wall_total,'train_tokens_s':a.end*CHUNK/max(train_sec,1e-9),'wall_tokens_s':a.end*CHUNK/max(wall_total,1e-9),'peak_vram_mb':peak,'best_val32k_loss':best,'final_val32k':final32,'final_exact512':final512,'gpu':torch.cuda.get_device_name(0),'parameters':model.parameter_report(),'checkpoint_start':str(BASE/'step0.pt'),'note':'10 logical epochs = 500 steps each; not 10 full passes over train.bin. Hybrid V3 uses exact local softmax over current+previous chunk and detached linear global state for older context.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('SUMMARY',json.dumps(summary),flush=True)
