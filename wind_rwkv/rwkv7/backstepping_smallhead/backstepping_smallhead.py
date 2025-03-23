import os, torch as th
from torch.utils.cpp_extension import load

CHUNK_LEN = 16

class RWKV7_smallhead(th.autograd.Function):
    @staticmethod
    def forward(ctx, q,w,k,v,a,b,s0):
        B,T,H,C = w.shape
        assert T%CHUNK_LEN == 0
        if not th.compiler.is_compiling():
            assert hasattr(th.ops.wind_backstepping_smallhead, 'forward'), 'Requires a load kernel from load_backstepping_smallhead(head_size)'
            assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
            assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
            assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
            assert list(s0.shape) == [B,H,C,C]
        B,T,H,C = w.shape
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.empty(B,H,T//CHUNK_LEN,C,C, dtype=th.float32,device=w.device)
        sa = th.empty(B,T,H,C, dtype=th.float32,device=w.device)
        th.ops.wind_backstepping_smallhead.forward(w,q,k,v,a,b, s0,y,s,sa,sT)
        ctx.save_for_backward(w,q,k,v,a,b,s,sa)
        return y,sT
    @staticmethod
    def backward(ctx, dy, dsT):
        w,q,k,v,a,b,s,sa = ctx.saved_tensors
        B,T,H,C = w.shape
        if not th.compiler.is_compiling():
            assert all(i.dtype==th.bfloat16 for i in [dy,dsT])
            assert all(i.is_contiguous() for i in [dy,dsT])

        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        th.ops.wind_backstepping_smallhead.backward(w,q,k,v,a,b, dy,s,sa,dsT, dw,dq,dk,dv,da,db,ds0)
        return dq,dw,dk,dv,da,db,ds0

def attn_backstepping_smallhead(r,w,k,v,a,b, s0 = None):
    B,T,H,C = w.shape
    if s0 is None: s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return RWKV7_smallhead.apply(r,w,k,v,a,b, s0)

def load_backstepping_smallhead(head_size):
    if hasattr(th.ops.wind_backstepping_smallhead, 'forward'): return
    CUDA_FLAGS = ['-res-usage', f'-D_C_={head_size}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    path = os.path.dirname(__file__)
    load(name="wind_backstepping_smallhead", sources=[os.path.join(path,'backstepping_smallhead.cu'), os.path.join(path,'backstepping_smallhead.cpp')], is_python_module=False, verbose=False, extra_cuda_cflags=CUDA_FLAGS)
    assert hasattr(th.ops.wind_backstepping_smallhead, 'forward')

def attn_backstepping_smallhead_wrap(r,w,k,v,a,b, head_size):
    B,T,HC = w.shape
    C = head_size
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return attn_backstepping_smallhead(w,r,k,v,a,b,s0)[0].view(B,T,HC)
