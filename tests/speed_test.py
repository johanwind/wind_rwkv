import argparse, triton, torch as th

parser = argparse.ArgumentParser()
parser.add_argument('--batchsz', type=int, default=8)
parser.add_argument('--modeldim', type=int, default=4096)
parser.add_argument('--headsz', type=int, default=64)
parser.add_argument('--seqlen', type=int, default=4096)
parser.add_argument('--alg', type=str, default='smallhead')
parser.add_argument('--forward', action=argparse.BooleanOptionalAction) # Forward pass only
cmd_args = parser.parse_args()

def gen_rwkv7_data():
    q,w,k,v,a,b = th.randn(6, cmd_args.batchsz, cmd_args.seqlen, cmd_args.modeldim//cmd_args.headsz, cmd_args.headsz, dtype = th.bfloat16, device = 'cuda')
    w = -th.nn.functional.softplus(w)-0.5
    a = th.nn.functional.normalize(a, p=2, dim=-1)
    b = -a*th.sigmoid(b)
    s0 = th.randn(cmd_args.batchsz, cmd_args.modeldim//cmd_args.headsz, cmd_args.headsz, cmd_args.headsz, dtype = th.bfloat16, device = 'cuda')
    return q,w,k,v,a,b,s0

def benchmark(f, params):
    if not cmd_args.forward:
        for p in params: p.requires_grad_()
    dy = ds = None
    def wrap():
        y,s = f(*params)
        if cmd_args.forward: return
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

params = gen_rwkv7_data()

if cmd_args.alg != 'fla':
    from wind_rwkv.rwkv7 import *
    if cmd_args.alg == 'smallhead':
        print('Backstepping smallhead fp32')
        load_backstepping_smallhead(cmd_args.headsz)
        benchmark(attn_backstepping_smallhead, params)
    elif cmd_args.alg == 'longhead':
        print('Backstepping longhead fp32')
        nheads = cmd_args.modeldim//cmd_args.headsz
        load_backstepping_longhead(cmd_args.headsz, cmd_args.batchsz * nheads)
        benchmark(attn_backstepping_longhead, params)
    elif cmd_args.alg == 'chunked':
        print('Chunked cuda')
        load_chunked_cuda(cmd_args.headsz)
        benchmark(attn_chunked_cuda, params)
    elif cmd_args.alg == 'bighead_bf16':
        print('Triton bighead bf16')
        benchmark(attn_triton_bighead_bf16, params)
    elif cmd_args.alg == 'bighead_fp16':
        print('Triton bighead fp16')
        benchmark(attn_triton_bighead_fp16, params)
    elif cmd_args.alg == 'bighead_fp32':
        print('Triton bighead fp32')
        benchmark(attn_triton_bighead_fp32, params)
    else:
        print('Unknown alg', cmd_args.arg)
else:
    assert cmd_args.alg == 'fla'
    print('FLA chunk_rwkv7')
    from fla.ops.rwkv7 import chunk_rwkv7
    def attn_fla(r,w,k,v,a,b,s0):
        return chunk_rwkv7(r,-w.exp(),k,v,a,b, initial_state=s0)
    benchmark(attn_fla, params)
