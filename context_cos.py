import json,math
from pathlib import Path
from collections import defaultdict
Q=Path('model/quality');ctx=defaultdict(lambda:[defaultdict(int),defaultdict(int)])
for line in (Q/'p2.jsonl').open(encoding='utf8'):
 d=json.loads(line);a,b=d['k'].split('\t');n=d['n'];ctx[a][1][b]+=n;ctx[b][0][a]+=n
def cos(a,b):
 A={('L',k):v for k,v in ctx[a][0].items()};A.update({('R',k):v for k,v in ctx[a][1].items()});B={('L',k):v for k,v in ctx[b][0].items()};B.update({('R',k):v for k,v in ctx[b][1].items()});dot=sum(v*B.get(k,0) for k,v in A.items());na=math.sqrt(sum(v*v for v in A.values()));nb=math.sqrt(sum(v*v for v in B.values()));return dot/(na*nb) if na*nb else 0,len(A),len(B)
for a,b in [('controlado','controlada'),('severo','severa'),('mínimo','mínima'),('significativo','significativa'),('elevado','elevada'),('secundário','secundária'),('moderado','moderada'),('alto','alta'),('baixo','baixa'),('casa','caso')]:print(a,b,cos(a,b))
