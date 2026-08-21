import json
from pathlib import Path
Q=Path('model/quality')
words={}
for line in (Q/'tokens.jsonl').open(encoding='utf8'):
 d=json.loads(line);w=d['k'];
 if any(st in w for st in ['controlad','sever','mínim','minim','significativ','elevad','secund','moderad','alt','baix']):words[w]=d['n']
print(json.dumps(dict(sorted(words.items(),key=lambda x:-x[1])[:100]),ensure_ascii=False,indent=2))
# show exact p2 involving target forms
forms={'controlado','controlada','severo','severa','mínimo','mínima','minimo','minima','significativo','significativa','elevado','elevada','secundário','secundária','moderado','moderada','alto','alta','baixo','baixa'}
hits=[]
for line in (Q/'p2.jsonl').open(encoding='utf8'):
 d=json.loads(line);parts=d['k'].split('\t')
 if any(x in forms for x in parts):hits.append((d['n'],d['k']))
print('P2',len(hits))
for x in sorted(hits,reverse=True)[:100]:print(x)
