import torch as th
import triton.testing

def naive(r,w,k,v,a,b,s0, inds):
    B,T,H,C = w.shape
    sT = th.empty_like(s0)
    w_dtype = w.dtype
    r,w,k,v,a,b = [i.double() for i in [r,w,k,v,a,b]]
    y = th.empty_like(v)
    for bi in range(B):
        for hi in range(H):
            for j in range(states_per_head):
                s = s0[bi,hi,j].double()
                for t in range(T):
                    if inds[bi,t,hi] == j:
                        s = s * th.exp(-th.exp(w[bi,t,hi,None,:])) + s @ a[bi,t,hi,:,None] * b[bi,t,hi,None,:] + v[bi,t,hi,:,None] * k[bi,t,hi,None,:]
                        y[bi,t,hi,:,None] = s @ r[bi,t,hi,:,None]
                sT[bi,hi,j] = s
    return y.to(w_dtype), sT

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

def benchmark(f, params, backward = True, aux=()):
    if backward:
        for p in params: p.requires_grad_()
    dy = ds = None
    def wrap():
        y,s = f(*params, *aux)
        if not backward: return
        nonlocal dy,ds
        if dy is None: dy,ds = th.randn_like(y),th.randn_like(s)
        return th.autograd.grad(y, params, grad_outputs=(dy,ds))

    wrap() # Warmup (compile triton)
    th.cuda.synchronize()
    th.cuda.reset_peak_memory_stats()
    wrap() # Measure memory
    th.cuda.synchronize()
    print(f'Peak VRAM {th.cuda.max_memory_allocated()/2**30:.2f} GB')
    ms, min_ms, max_ms = triton.testing.do_bench(wrap, quantiles=[0.5,0.2,0.8], warmup=1000,rep=2000)
    print('Time', f'{ms:.2f} ms ({min_ms:.2f} - {max_ms:.2f})')


batchsz = 2
modeldim = 1024
headsz = 64
seqlen = 128
states_per_head = 4

def gen_rwkv7_data():
    q,w,k,v,a,b = th.randn(6, batchsz, seqlen, modeldim//headsz, headsz, dtype = th.bfloat16, device = 'cuda')
    w = -th.nn.functional.softplus(w)-0.5
    a = th.nn.functional.normalize(a, p=2, dim=-1)
    b = -a*th.sigmoid(b)
    s0 = th.randn(batchsz, modeldim//headsz, states_per_head, headsz, headsz, dtype = th.bfloat16, device = 'cuda')
    inds = th.randint(0, states_per_head, (batchsz, seqlen, modeldim//headsz), dtype = th.int, device = 'cuda')
    return q,w,k,v,a,b,s0,inds

th.manual_seed(0)
params = gen_rwkv7_data()

from wind_rwkv.experimental import load_smallhead_moh, attn_smallhead_moh

load_smallhead_moh(headsz, states_per_head)
grad_check(attn_smallhead_moh, naive, params[:-1], aux=params[-1:])

batchsz = 8
modeldim = 4096
headsz = 64
seqlen = 1024

params = gen_rwkv7_data()

benchmark(attn_smallhead_moh, params[:-1], aux=params[-1:])
