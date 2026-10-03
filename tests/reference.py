"""Run after node tests/verify.cjs; verify JS probabilities against NumPy."""
import json,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import norm,softmax
page=Path(__file__).resolve().parents[1].joinpath('index.html').read_text()
m=json.loads(re.search(r'<script id="model-weights" type="application/json">(.*?)</script>',page,re.S)[1])
p={k:np.array(v['data'],dtype=float).reshape(v['shape'])*v['scale'] for k,v in m['weights'].items()}
text='Alice was beginning '
ids=[m['vocab'].index(c) for c in text]; T=len(ids)
x=p['emb'][ids]+p['pos'][:T]
for l in range(2):
    n,_=norm(x)
    q,k,v=(n@p[f'{l}.qkv']).reshape(T,3,2,12).transpose(1,2,0,3)
    a=softmax(np.where(np.triu(np.ones((T,T),dtype=bool),1),-1e9,q@k.swapaxes(-1,-2)/np.sqrt(12)))
    c=(a@v).transpose(1,0,2).reshape(T,24)
    x=x+c@p[f'{l}.o']; n,_=norm(x)
    x=x+np.maximum(0,n@p[f'{l}.fc1'])@p[f'{l}.fc2']
n,_=norm(x); probs=softmax(n[-1]@p['head']+p['bias'])
js=np.array(json.loads(Path('/private/tmp/pocket-js-probs.json').read_text()))
assert np.allclose(probs,js,rtol=1e-10,atol=1e-12),np.max(np.abs(probs-js))
print('PASS: NumPy / JS max absolute probability error:',np.max(np.abs(probs-js)))
