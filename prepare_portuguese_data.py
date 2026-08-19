from datasets import load_dataset
from pathlib import Path
import json, time, shutil
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data'
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()
# Keep total corpus <= 2GB on disk: 900MiB train + 50MiB val + 50MiB test = 1000MiB.
TARGETS={'train':900*1024**2,'val':50*1024**2,'test':50*1024**2}
ds=load_dataset('HuggingFaceFW/fineweb-2', name='por_Latn', streaming=True)

def write_stream(stream, specs):
    files={k:(OUT/f'{k}.bin').open('wb') for k,_ in specs}
    counts={k:0 for k,_ in specs}; docs={k:0 for k,_ in specs}
    keys=[k for k,_ in specs]; idx=0
    for row in stream:
        if idx>=len(specs): break
        k,target=specs[idx]
        b=(row['text'].strip()+'\n\n').encode('utf-8', errors='ignore')
        remain=target-counts[k]
        if remain<=0:
            idx+=1; continue
        if len(b)>remain: b=b[:remain]
        files[k].write(b); counts[k]+=len(b); docs[k]+=1
        if counts[k]>=target:
            files[k].flush(); idx+=1
    for f in files.values(): f.close()
    return counts,docs
start=time.time()
counts1,docs1=write_stream(ds['train'], [('train',TARGETS['train']),('val',TARGETS['val'])])
counts2,docs2=write_stream(ds['test'], [('test',TARGETS['test'])])
meta={'dataset':'HuggingFaceFW/fineweb-2','config':'por_Latn','tokenizer':'utf8-bytes','vocab_size':256,'targets_bytes':TARGETS,'actual_bytes':{**counts1,**counts2},'documents':{**docs1,**docs2},'elapsed_sec':time.time()-start}
(OUT/'meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(meta,indent=2,ensure_ascii=False))

