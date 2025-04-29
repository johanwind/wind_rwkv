import torch as th
from wind_rwkv.rwkv7 import *

def naive(r,w,k,v,a,b,s0, cu_seqlens):
    T,H,C = r.shape
    B = len(cu_seqlens)-1
    if s0 is None: s0 = th.zeros(B,H,C,C, device=w.device)
    dtype = w.dtype
    r,w,k,v,a,b,s0 = [i.double() for i in [r,w,k,v,a,b,s0]]
    y = th.empty_like(v)
    sT = th.empty_like(s0)
    bi = 0
    for t in range(T):
        if t == cu_seqlens[bi]:
            s = s0[bi]
            bi += 1
        s = s * th.exp(-th.exp(w[t,:,None,:])) + s @ a[t,:,:,None] * b[t,:,None,:] + v[t,:,:,None] * k[t,:,None,:]
        y[t,:,:,None] = s @ r[t,:,:,None]
        if t+1 == cu_seqlens[bi]:
            sT[bi-1] = s
    return y.to(dtype), sT.to(dtype)

def grad_check(f1, f2, params, backward = True, aux=()):
    if backward: params = [p.clone().requires_grad_() for p in params]
    y1 = f1(*params,*aux)
    y2 = f2(*params,*aux)
    def rel(a,b): return (a-b).norm()/max(b.norm(),1e-30)
    print('Forward rel. error'+'s'*(len(y1)>1))
    for a,b in zip(y1,y2):
        print(f'{rel(a,b):.2e}  ({b.norm():.0e})')

    if not backward: return

    dy = tuple(th.randn_like(i) for i in y1)
    d1 = th.autograd.grad(y1, params, grad_outputs=dy)
    for p in params:
        if p.grad is not None:
            p.grad.random_() # So th.empty doesn't recover the gradient
        p.grad = None
    d2 = th.autograd.grad(y2, params, grad_outputs=dy)
    print('Gradient rel. errors')
    for a,b in zip(d1,d2):
        print(f'{rel(a,b):.2e}  ({b.norm():.0e})')

cu_seqlens = th.tensor([0,16,48,64], device='cuda')
modeldim = 1024
headsz = 128

def gen_rwkv7_data():
    q,w,k,v,a,b = th.randn(6, cu_seqlens[-1], modeldim//headsz, headsz, dtype = th.bfloat16, device = 'cuda')
    w = -th.nn.functional.softplus(w)-0.5
    a = th.nn.functional.normalize(a, p=2, dim=-1)
    b = -a*th.sigmoid(b)
    s0 = th.randn(len(cu_seqlens)-1, modeldim//headsz, headsz, headsz, dtype = th.bfloat16, device = 'cuda')
    return q,w,k,v,a,b,s0

th.manual_seed(0)
params = gen_rwkv7_data()

if 0:
    print('FLA chunk_rwkv7')
    from fla.ops.rwkv7 import chunk_rwkv7
    def attn_fla(r,w,k,v,a,b,s0, cu_seqlens):
        r,w,k,v,a,b = [i.unsqueeze(0) for i in [r,w,k,v,a,b]]
        y,sT = chunk_rwkv7(r,-w.exp(),k,v,a,b, initial_state=s0.mT, cu_seqlens=cu_seqlens)
        return y.squeeze(0), sT.mT
    grad_check(attn_fla, naive, params, aux=(cu_seqlens,))

print('Chunked cuda varlen')
load_chunked_cuda_varlen(headsz)
grad_check(attn_chunked_cuda_varlen, naive, params, aux=(cu_seqlens,))
