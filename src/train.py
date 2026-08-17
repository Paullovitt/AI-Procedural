import argparse,json,math,time,random
from pathlib import Path
import numpy as np, torch
from model import ModelConfig,TinyTransformer

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_batch(data,bs,seq,device):
    starts=np.random.randint(0,len(data)-seq-1,size=bs)
    x=np.stack([np.asarray(data[s:s+seq],dtype=np.int64) for s in starts])
    y=np.stack([np.asarray(data[s+1:s+seq+1],dtype=np.int64) for s in starts])
    return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)

@torch.no_grad()
def evaluate(model,data,bs,seq,device,amp,batches=10,seed=777):
    state=np.random.get_state(); np.random.seed(seed); model.eval(); vals=[]
    for _ in range(batches):
        x,y=get_batch(data,bs,seq,device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            _,loss=model(x,y)
        vals.append(float(loss.item()))
    model.train(); np.random.set_state(state)
    v=sum(vals)/len(vals); return v,math.exp(min(20,v))

def lr_at(step,total,peak,min_lr,warmup):
    if step < warmup: return peak*(step+1)/max(1,warmup)
    if total <= warmup: return min_lr
    p=(step-warmup)/max(1,total-warmup)
    return min_lr+0.5*(peak-min_lr)*(1+math.cos(math.pi*min(1.0,p)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['baseline','procedural'],default='procedural')
    ap.add_argument('--out-dir',default='runs/procedural80m')
    ap.add_argument('--d-model',type=int,default=744)
    ap.add_argument('--layers',type=int,default=12)
    ap.add_argument('--heads',type=int,default=12)
    ap.add_argument('--rank',type=int,default=64)
    ap.add_argument('--ffn-mult',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--grad-accum',type=int,default=16)
    ap.add_argument('--seq-len',type=int,default=256)
    ap.add_argument('--steps',type=int,default=115200)
    ap.add_argument('--logical-epochs',type=int,default=10)
    ap.add_argument('--resume',default='')
    ap.add_argument('--run-until',type=int,default=0)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--min-lr',type=float,default=2e-5)
    ap.add_argument('--warmup',type=int,default=1000)
    ap.add_argument('--eval-interval',type=int,default=1000)
    ap.add_argument('--eval-batches',type=int,default=10)
    ap.add_argument('--checkpoint',action='store_true')
    ap.add_argument('--no-fused-qkv',action='store_true')
    ap.add_argument('--seed',type=int,default=1337)
    a=ap.parse_args(); seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp=device.type=='cuda'
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    tr=np.memmap('data/train.bin',dtype=np.uint8,mode='r'); va=np.memmap('data/val.bin',dtype=np.uint8,mode='r')
    cfg=ModelConfig(vocab_size=256,d_model=a.d_model,n_layers=a.layers,n_heads=a.heads,seq_len=a.seq_len,ffn_mult=a.ffn_mult,procedural_rank=a.rank,model_type=a.model,activation_checkpoint=a.checkpoint,fused_qkv=not a.no_fused_qkv)
    model=TinyTransformer(cfg).to(device)
    try:
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.1,fused=(device.type=='cuda'))
    except TypeError:
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.1)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    effective_tokens=a.batch_size*a.seq_len*a.grad_accum
    steps_per_pass=math.ceil((len(tr)-1)/effective_tokens)
    dense_equivalent=79936848 if (a.d_model,a.layers,a.heads,a.ffn_mult)==(744,12,12,4) else None
    conf={'args':vars(a),'model_config':cfg.__dict__,'params':model.parameter_report(),'dense_equivalent_params':dense_equivalent,'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU','effective_tokens_per_step':effective_tokens,'dataset_steps_per_pass':steps_per_pass,'dataset_bytes':len(tr)}
    (out/'config.json').write_text(json.dumps(conf,indent=2),encoding='utf-8')
    if not a.resume:
        torch.save({'model':model.state_dict(),'config':conf,'step':0},out/'step0.pt')
    start_step=1; best=1e9
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(ck['model']); start_step=int(ck.get('step',0))+1; best=float(ck.get('val_loss',1e9))
        if 'optimizer' in ck: opt.load_state_dict(ck['optimizer'])
        if 'scaler' in ck: scaler.load_state_dict(ck['scaler'])
    t0=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    model.train()
    end_step=min(a.steps,a.run_until) if a.run_until>0 else a.steps
    for step in range(start_step,end_step+1):
        lr=lr_at(step-1,a.steps,a.lr,a.min_lr,a.warmup)
        for g in opt.param_groups: g['lr']=lr
        opt.zero_grad(set_to_none=True); ls=0.0
        for _ in range(a.grad_accum):
            x,y=get_batch(tr,a.batch_size,a.seq_len,device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                _,loss=model(x,y); loss=loss/a.grad_accum
            scaler.scale(loss).backward(); ls+=float(loss.item())
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        if step%a.eval_interval==0 or step==end_step:
            vl,ppl=evaluate(model,va,min(a.batch_size,4),a.seq_len,device,amp,a.eval_batches)
            elapsed=time.perf_counter()-t0; toks=step*effective_tokens
            peak=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0.0
            r={'step':step,'train_loss':ls,'val_loss':vl,'val_ppl':ppl,'lr':lr,'peak_vram_mb':peak,'tokens_s_avg':toks/elapsed,'tokens_seen':toks,'dataset_passes':toks/len(tr)}
            print(json.dumps(r),flush=True)
            ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'config':conf,'step':step,'val_loss':vl,'logical_epoch':min(a.logical_epochs,math.ceil(step/(a.steps/a.logical_epochs)))}; torch.save(ck,out/'last.pt')
            if vl<best: best=vl; torch.save(ck,out/'best.pt')
    summary={'best_val_loss':best,'best_val_ppl':math.exp(min(20,best)),'params':model.parameter_report(),'dense_equivalent_params':dense_equivalent,'steps_completed':end_step,'target_steps':a.steps,'logical_epochs':a.logical_epochs,'effective_tokens_per_step':effective_tokens,'tokens_seen':end_step*effective_tokens,'dataset_passes':end_step*effective_tokens/len(tr)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__': main()





