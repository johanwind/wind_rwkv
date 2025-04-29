import os, torch as th
from torch.utils.cpp_extension import load

class RWKV7_chunked_varlen(th.autograd.Function):
    @staticmethod
    def forward(ctx, q,w,k,v,a,b,s0, cu_seqlens):
        T,H,C = w.shape
        assert T%16 == 0
        if not th.compiler.is_compiling():
            assert hasattr(th.ops.wind_chunked_cuda_varlen, 'forward'), 'Requires a load kernel from load_chunked_cuda_varlen(head_size)'
            assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b,s0])
            assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
            assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
            assert list(s0.shape) == [len(cu_seqlens)-1,H,C,C]
            assert cu_seqlens.dtype == th.long
            assert cu_seqlens.device == w.device
            assert (cu_seqlens%16 == 0).all()
            assert cu_seqlens[-1].item() == T
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        if any(i.requires_grad for i in [w,q,k,v,a,b,s0]):
            s = th.empty(H,T//16,C,C, dtype=th.bfloat16,device=w.device)
        else:
            s = None
        th.ops.wind_chunked_cuda_varlen.forward(w,q,k,v,a,b, s0,y,s,sT, cu_seqlens)
        ctx.save_for_backward(w,q,k,v,a,b,s,cu_seqlens)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        w,q,k,v,a,b,s,cu_seqlens = ctx.saved_tensors
        if not th.compiler.is_compiling():
            assert all(i.dtype==th.bfloat16 for i in [dy,dsT])
            assert all(i.is_contiguous() for i in [dy,dsT])
        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        th.ops.wind_chunked_cuda_varlen.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0, cu_seqlens)
        return dq,dw,dk,dv,da,db,ds0,None

def attn_chunked_cuda_varlen(r,w,k,v,a,b,s0,cu_seqlens):
    T,H,C = w.shape
    B = len(cu_seqlens)-1
    if s0 is None: s0 = th.zeros(B,H,C,C, dtype=th.bfloat16,device=w.device)
    return RWKV7_chunked_varlen.apply(r,w,k,v,a,b,s0,cu_seqlens)

def load_chunked_cuda_varlen(head_size):
    if hasattr(th.ops.wind_chunked_cuda_varlen, 'forward'): return
    CUDA_FLAGS = ["-res-usage", f'-D_C_={head_size}', "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    if head_size == 256: CUDA_FLAGS.append('-maxrregcount=128')
    path = os.path.dirname(__file__)
    load(name="wind_chunked_cuda_varlen", sources=[os.path.join(path,'chunked_cuda_varlen.cu'), os.path.join(path,'chunked_cuda_varlen.cpp')], is_python_module=False, verbose=False, extra_cuda_cflags=CUDA_FLAGS)
    assert hasattr(th.ops.wind_chunked_cuda_varlen, 'forward')
