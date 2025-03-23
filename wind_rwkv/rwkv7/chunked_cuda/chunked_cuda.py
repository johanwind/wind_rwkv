import os, torch as th
from torch.utils.cpp_extension import load

class RWKV7_chunked(th.autograd.Function):
    @staticmethod
    def forward(ctx, q,w,k,v,a,b,s0):
        B,T,H,C = w.shape
        assert T%16 == 0
        if not th.compiler.is_compiling():
            assert hasattr(th.ops.wind_chunked_cuda, 'forward'), 'Requires a load kernel from load_chunked_cuda(head_size)'
            assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
            assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
            assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
            assert list(s0.shape) == [B,H,C,C]
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        if any(i.requires_grad for i in [w,q,k,v,a,b,s0]):
            s = th.empty(B,H,T//16,C,C, dtype=th.bfloat16,device=w.device)
        else:
            s = None
        th.ops.wind_chunked_cuda.forward(w,q,k,v,a,b, s0,y,s,sT)
        ctx.save_for_backward(w,q,k,v,a,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        w,q,k,v,a,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        if not th.compiler.is_compiling():
            assert all(i.dtype==th.bfloat16 for i in [dy,dsT])
            assert all(i.is_contiguous() for i in [dy,dsT])
        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        th.ops.wind_chunked_cuda.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
        return dq,dw,dk,dv,da,db,ds0

def attn_chunked_cuda(r,w,k,v,a,b, s0 = None):
    B,T,H,C = w.shape
    if s0 is None: s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return RWKV7_chunked.apply(r,w,k,v,a,b, s0)

def load_chunked_cuda(head_size):
    if hasattr(th.ops.wind_chunked_cuda, 'forward'): return
    CUDA_FLAGS = ["-res-usage", f'-D_C_={head_size}', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    if head_size == 256: CUDA_FLAGS.append('-maxrregcount=128')
    path = os.path.dirname(__file__)
    load(name="wind_chunked_cuda", sources=[os.path.join(path,'chunked_cuda.cu'), os.path.join(path,'chunked_cuda.cpp')], is_python_module=False, verbose=False, extra_cuda_cflags=CUDA_FLAGS)
    assert hasattr(th.ops.wind_chunked_cuda, 'forward')

def attn_chunked_cuda_wrap(r,w,k,v,a,b, head_size):
    B,T,HC = w.shape
    C = head_size
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return WindRWKV7.apply(w,r,k,v,a,b,s0)[0].view(B,T,HC)
