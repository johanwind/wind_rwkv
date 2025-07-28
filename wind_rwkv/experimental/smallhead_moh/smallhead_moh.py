import os, torch as th
from torch.utils.cpp_extension import load

CHUNK_LEN = 16

class RWKV7_smallhead_moh(th.autograd.Function):
    @staticmethod
    def forward(ctx, q,w,k,v,a,b,s0, inds):
        B,T,H,C = w.shape
        if not th.compiler.is_compiling():
            assert hasattr(th.ops.wind_smallhead_moh, 'forward'), 'Requires a loaded kernel from load_smallhead_moh(head_size)'
            assert all(i.dtype==th.bfloat16 for i in [w,q,k,v,a,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0,inds])
            assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
            assert list(s0.shape) == [B,H,states_per_head,C,C]
            assert list(inds.shape) == [B,T,H]
        s0 = s0.float()
        inds = inds.int()
        B,T,H,C = w.shape
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.empty(B,H,(T-1)//CHUNK_LEN+1,C,C, dtype=th.float32,device=w.device)
        sa = th.empty(B,T,H,C, dtype=th.float32,device=w.device)

        cnts = (inds.unsqueeze(-1)==th.arange(states_per_head,dtype=th.int,device=w.device)).sum(1).cumsum(2).int() # [B,H,K]

        th.ops.wind_smallhead_moh.forward(w,q,k,v,a,b, s0,y,s,sa,sT, inds,cnts)
        ctx.save_for_backward(w,q,k,v,a,b,s,sa,sT,inds,cnts)
        return y,sT
    @staticmethod
    def backward(ctx, dy, dsT):
        w,q,k,v,a,b,s,sa,sT,inds,cnts = ctx.saved_tensors
        B,T,H,C = w.shape
        if not th.compiler.is_compiling():
            assert dy.dtype == th.bfloat16
            assert all(i.is_contiguous() for i in [dy,dsT])
        dsT = dsT.float()

        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        th.ops.wind_smallhead_moh.backward(w,q,k,v,a,b, dy,s,sa,dsT, dw,dq,dk,dv,da,db,ds0, sT,inds,cnts)
        return dq,dw,dk,dv,da,db,ds0, None

def attn_smallhead_moh(r,w,k,v,a,b,s0, inds):
    B,T,H,C = w.shape
    if s0 is None: s0 = th.zeros(B,H,states_per_head,C,C, dtype=th.float32,device=w.device)
    return RWKV7_smallhead_moh.apply(r,w,k,v,a,b, s0, inds)

def load_smallhead_moh(head_size, states_per_head_):
    if hasattr(th.ops.wind_smallhead_moh, 'forward'): return
    global states_per_head
    states_per_head = states_per_head_
    device_props = th.cuda.get_device_properties(th.cuda.current_device())
    CUDA_FLAGS = ['-res-usage', f'-D_K_={states_per_head}', f'-D_C_={head_size}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    path = os.path.dirname(__file__)
    load(name="wind_smallhead_moh", sources=[os.path.join(path,'smallhead_moh.cu'), os.path.join(path,'smallhead_moh.cpp')], is_python_module=False, verbose=False, extra_cuda_cflags=CUDA_FLAGS)
    assert hasattr(th.ops.wind_smallhead_moh, 'forward')
