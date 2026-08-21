from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parent
p=ROOT/'rigorous_results'/'position_auto_calibration.json'
d=json.loads(p.read_text(encoding='utf8'));rows=d['rows'];base=d['baseline']
valid=[x for x in rows if x['bad']==0 and x['slot_errors']==0 and x['support']>=base['support']-.005 and x['repeat']<=base['repeat']+.02]
best=max(valid,key=lambda x:x['contrast_win_rate'])
# Wilson 95% lower confidence bound of the empirically best contrast rate.
n=best['comparisons'];ph=best['contrast_win_rate'];z=1.959963984540054
center=(ph+z*z/(2*n))/(1+z*z/n)
half=z*math.sqrt((ph*(1-ph)+z*z/(4*n))/n)/(1+z*z/n)
lower=center-half
# Minimum-complexity equivalent: smallest weight whose observed performance is
# inside the best model's 95% confidence region, under the hard quality gates.
elig=[x for x in valid if x['contrast_win_rate']>=lower]
selected=min(elig,key=lambda x:(x['position_weight'],-x['support'],x['repeat']))
d['selection_method']='minimum position weight inside 95% Wilson lower bound of best contrast, under semantic/support/repetition gates'
d['best_observed']=best;d['best_wilson95_lower']=lower;d['selected']=selected
p.write_text(json.dumps(d,indent=2),encoding='utf8')
print(json.dumps({'best':best,'wilson95_lower':lower,'selected':selected},indent=2))
