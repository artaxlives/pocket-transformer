"""Offline NumPy trainer. Python 3 + numpy; no framework or network needed.
Run: python3 train.py --steps 3000 > weights.json
Replace corpus.txt with UTF-8 text to retrain; paste stdout JSON into index.html's
model-weights script. Progress goes to stderr. Seed 42 reproduces initialization
and batch selection. Architecture: pre-norm, 2 layers, 2 heads, width 24,
48 positions, ReLU MLP width 48, fixed (non-affine) layer normalization.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

D, T, H = 24, 48, 2

def norm(x):
    z = x - x.mean(-1, keepdims=True)
    inv = 1 / np.sqrt((z*z).mean(-1, keepdims=True) + 1e-5)
    return z*inv, inv

def norm_back(g, z, inv):
    return inv*(g-g.mean(-1,keepdims=True)-z*(g*z).mean(-1,keepdims=True))

def softmax(x):
    e = np.exp(x-x.max(-1,keepdims=True))
    return e/e.sum(-1,keepdims=True)

def run(steps):
    rng = np.random.default_rng(42)
    corpus = Path(__file__).with_name('corpus.txt').read_text()
    vocab = ''.join(sorted(set(corpus + '0123456789!?;:.,\"\'()-')))
    ids = np.array([vocab.index(c) for c in corpus])
    V = len(vocab)
    p = {'emb': rng.normal(0,.15,(V,D)), 'pos':rng.normal(0,.03,(T,D))}
    for l in range(2):
        for name, shape in [('qkv',(D,3*D)),('o',(D,D)),('fc1',(D,2*D)),('fc2',(2*D,D))]:
            p[f'{l}.{name}']=rng.normal(0,.13,shape)
    p['head']=rng.normal(0,.13,(D,V)); p['bias']=np.zeros(V)
    p={k:v.astype(np.float32) for k,v in p.items()}
    m={k:np.zeros_like(v) for k,v in p.items()}; v={k:np.zeros_like(v) for k,v in p.items()}
    B=12
    mask=np.triu(np.ones((T,T),dtype=bool),1)
    for step in range(1,steps+1):
        starts=rng.integers(0,len(ids)-T-1,B)
        ix=ids[starts[:,None]+np.arange(T)]; target=ids[starts[:,None]+np.arange(1,T+1)]
        x=p['emb'][ix]+p['pos']; cache=[]
        for l in range(2):
            n, ni=norm(x)
            q,k,val=(n@p[f'{l}.qkv']).reshape(B,T,3,H,D//H).transpose(2,0,3,1,4)
            a=softmax(np.where(mask,-1e9,(q@k.swapaxes(-1,-2))/np.sqrt(D//H)))
            c=(a@val).transpose(0,2,1,3).reshape(B,T,D)
            y=x+c@p[f'{l}.o']; z,zi=norm(y)
            r=np.maximum(0,z@p[f'{l}.fc1']); x=y+r@p[f'{l}.fc2']
            cache.append((n,ni,q,k,val,a,c,z,zi,r))
        f,fi=norm(x); prob=softmax(f@p['head']+p['bias'])
        loss=-np.log(prob[np.arange(B)[:,None],np.arange(T),target]+1e-12).mean()
        g=prob.copy(); g[np.arange(B)[:,None],np.arange(T),target]-=1; g/=B*T
        grad={'head':f.reshape(-1,D).T@g.reshape(-1,V),'bias':g.sum((0,1))}
        dx=norm_back(g@p['head'].T,f,fi)
        for l in reversed(range(2)):
            n,ni,q,k,val,a,c,z,zi,r=cache[l]
            grad[f'{l}.fc2']=r.reshape(-1,2*D).T@dx.reshape(-1,D)
            dr=(dx@p[f'{l}.fc2'].T)*(r>0)
            grad[f'{l}.fc1']=z.reshape(-1,D).T@dr.reshape(-1,2*D)
            dy=dx+norm_back(dr@p[f'{l}.fc1'].T,z,zi)
            grad[f'{l}.o']=c.reshape(-1,D).T@dy.reshape(-1,D)
            dc=(dy@p[f'{l}.o'].T).reshape(B,T,H,D//H).transpose(0,2,1,3)
            da=dc@val.swapaxes(-1,-2); dv=a.swapaxes(-1,-2)@dc
            ds=a*(da-(da*a).sum(-1,keepdims=True))/np.sqrt(D//H)
            dq=ds@k; dk=ds.swapaxes(-1,-2)@q
            dqkv=np.stack([dq,dk,dv]).transpose(1,3,0,2,4).reshape(B,T,3*D)
            grad[f'{l}.qkv']=n.reshape(-1,D).T@dqkv.reshape(-1,3*D)
            dx=dy+norm_back(dqkv@p[f'{l}.qkv'].T,n,ni)
        grad['pos']=dx.sum(0); grad['emb']=np.zeros_like(p['emb']); np.add.at(grad['emb'],ix,dx)
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grad.values()):
            raise FloatingPointError(f'Non-finite loss or gradients at step {step}')
        lr=.003*(.2+.8*(1-step/steps))
        for key in p:
            gg=np.clip(grad[key],-1,1)
            m[key]=.9*m[key]+.1*gg; v[key]=.999*v[key]+.001*gg*gg
            p[key]-=lr*(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.999**step))+1e-8)
        if step==1 or step%250==0: print(f'{step}: loss {loss:.4f}',file=sys.stderr)
    weights={}
    for key,w in p.items():
        scale=float(np.max(np.abs(w))/127) or 1
        weights[key]={'shape':list(w.shape),'scale':scale,'data':np.rint(w/scale).astype(int).ravel().tolist()}
    return {'vocab':vocab,'d':D,'context':T,'heads':H,'layers':2,'steps':steps,'seed':42,'loss':float(loss),'weights':weights}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=3000)
    args=parser.parse_args()
    if args.steps<1: parser.error('--steps must be positive')
    print(json.dumps(run(args.steps),separators=(',',':')))
