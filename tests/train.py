# Copyright (c) 2025, Johan Sokrates Wind

# Based on https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp

"""
# Install wind_rwkv (from root wind_rwkv directory)
pip install -e .

# Install requirements
pip install deepspeed ninja wandb

# Download data, we use minipile (1498226207 tokens, around 3GB)
mkdir -p data
wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin

# Run on a single gpu with ~8GB RAM
torchrun train.py --micro_bsz 12
# Run on 4 gpus without gradient checkpointing
torchrun --nproc-per-node=4 train.py --grad_cp 0

# First run creates out/L12-D768/rwkv-init.pth, subsequent runs will continue from latest checkpoint in out/L12-D768/

# out/L12-D768/train_log.txt losses should be similar to
0 5.056944 157.1097 0.00059976 2025-06-09 00:01:03.903503
1 4.016493 55.5061 0.00059901 2025-06-09 00:04:43.806584
2 3.750670 42.5496 0.00059775 2025-06-09 00:08:23.082158
3 3.630432 37.7291 0.00059600 2025-06-09 00:12:02.702873
4 3.553571 34.9379 0.00059374 2025-06-09 00:15:42.810617
5 3.486632 32.6757 0.00059099 2025-06-09 00:19:22.601675
6 3.434177 31.0059 0.00058775 2025-06-09 00:23:01.326041
7 3.381845 29.4250 0.00058403 2025-06-09 00:26:41.194270
8 3.338046 28.1640 0.00057984 2025-06-09 00:30:21.751441
9 3.293576 26.9390 0.00057517 2025-06-09 00:34:01.008750
...
"""

import os, struct, math, tqdm, datetime, time, argparse

import torch
from torch import nn
import torch.nn.functional as F

import deepspeed, logging
deepspeed.utils.logger.setLevel(logging.WARNING)

from wind_rwkv.rwkv7 import load_chunked_cuda, attn_chunked_cuda


# Parse arguments

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", default="data/minipile", type=str)
    parser.add_argument("--proj_dir", default="auto", type=str)
    parser.add_argument("--wandb", default="Test", type=str)  # wandb project name. if "" then don't use wandb

    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--n_layer", default=12, type=int)
    parser.add_argument("--n_embd", default=768, type=int)
    parser.add_argument("--head_size", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--dim_ffn", default=0, type=int)

    parser.add_argument("--micro_bsz", default=16, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--ctx_len", default=512, type=int)

    parser.add_argument("--epoch_save", default=10, type=int)  # save the model every [epoch_save] "epochs"
    parser.add_argument("--samples_per_epoch", default=40320, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=6e-5, type=float)
    parser.add_argument("--warmup_steps", default=10, type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument("--grad_cp", default=1, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--weight_decay", default=1e-3, type=float) # try 0.1
    parser.add_argument("--grad_clip", default=1.0, type=float) # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples

    parser.add_argument("--torch_compile", default=1, type=int)
    parser.add_argument("--ds_bucket_mb", default=2, type=int)  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.proj_dir == "auto":
        args.proj_dir = f"out/L{args.n_layer}-D{args.n_embd}"
    if not args.dim_ffn:
        args.dim_ffn = args.n_embd*4

    assert all(i%32 == 0 for i in [args.n_embd, args.dim_ffn])

    args.global_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    args.total_bsz = args.micro_bsz * args.world_size
    assert args.samples_per_epoch % args.total_bsz == 0

    args.timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

    return args


# Model definition

def new_param(*shape): return nn.Parameter(torch.empty(*shape))

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args):
        super().__init__()
        C,N = args.n_embd, args.head_size
        self.n_head = C//N

        for p in "x_r x_w x_k x_v x_a x_g".split():
            setattr(self, p, new_param(1,1,C))

        dims = [max(32, 32*round(fac*C**p/32)) for fac, p in zip([1.8,1.8,1.3,0.6], [0.5,0.5,0.5,0.8])]
        for c, D in zip("wavg", dims):
            setattr(self, f"{c}1", new_param(C,D))
            setattr(self, f"{c}2", new_param(D,C))
            if c != "g":
                setattr(self, f"{c}0", new_param(1,1,C))

        self.k_k = new_param(1,1,C)
        self.k_a = new_param(1,1,C)
        self.r_k = new_param(C//N,N)

        self.receptance, self.key, self.value, self.output = [nn.Linear(C, C, bias=False) for i in range(4)]
        self.ln_x = nn.GroupNorm(C//N, C, eps=64e-5)

        load_chunked_cuda(args.head_size)

    def forward(self, x, v0):
        B,T,C = x.shape
        H = self.n_head

        last_x = F.pad(x, (0,0,1,-1))
        xr,xw,xk,xv,xa,xg = [x + m * (last_x - x) for m in [self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g]]

        r = self.receptance(xr)
        w = -F.softplus(-self.w0 - (xw @ self.w1).tanh() @ self.w2) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if v0 is None:
            v0 = v # store first layer's v
        else:
            v = v + (v0 - v) * (self.v0 + xv @ self.v1 @ self.v2).sigmoid()
        a = (self.a0 + xa @ self.a1 @ self.a2).sigmoid()
        g = (xg @ self.g1).sigmoid() @ self.g2

        kk = k * self.k_k
        k = k * (1 + (a-1) * self.k_a)

        r,w,k,v,kk,a = [i.reshape(B,T,H,-1) for i in [r,w,k,v,kk,a]]

        kk = F.normalize(kk, dim=-1)
        x = attn_chunked_cuda(r, w, k, v, kk, -kk*a)[0]
        x = self.ln_x(x.view(B*T, C)).view(B,T,C)

        x = x + ((r * k * self.r_k).sum(-1,True) * v).view(B,T,C)
        x = self.output(x * g)
        return x, v0

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_k = new_param(1,1,args.n_embd)
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        k = x + (F.pad(x, (0,0,1,-1)) - x) * self.x_k
        return self.value(self.key(k).relu()**2)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x070(args)
        self.ffn = RWKV_CMix_x070(args)

    def forward(self, x, v0):
        x_attn, v0 = self.att(self.ln1(x), v0)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))
        return x, v0

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args) for i in range(args.n_layer)])
        self.blocks[0].ln0 = nn.LayerNorm(args.n_embd)
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, tokens):
        x = self.blocks[0].ln0(self.emb(tokens))
        v0 = None
        for block in self.blocks:
            if args.grad_cp:
                x, v0 = deepspeed.checkpointing.checkpoint(block, x, v0)
            else:
                x, v0 = block(x, v0)
        return self.head(self.ln_out(x))


# Sample initial weights

def sample_initial_weights(model, args):
    W = model.state_dict()

    scale = 0.5*max(args.vocab_size / args.n_embd, 1)**0.5
    nn.init.orthogonal_(W["head.weight"], gain=scale)
    nn.init.uniform_(W["emb.weight"], a=-1e-4, b=1e-4)

    L,C,N = args.n_layer, args.n_embd, args.head_size
    for i in range(L):
        n = torch.arange(C)

        ffn = f"blocks.{i}.ffn."
        W[ffn+"x_k"][:] = 1-(n/C)**((1-i/L)**4)
        nn.init.orthogonal_(W[ffn+"key.weight"])
        W[ffn+"value.weight"][:] = 0

        att = f"blocks.{i}.att."
        for c,p in zip("rwkvag", [0.2,0.9,0.7,0.7,0.9,0.2]):
            W[att+"x_"+c][:] = 1-(n/C)**(p*(1-i/L))

        linear = n/(C-1)-0.5
        zigzag = (z := (n%N)*2/(N-1)-1) * z.abs()
        W[att+"k_k"][:] = 0.71 - linear*0.1
        W[att+"k_a"][:] = 1.02
        W[att+"r_k"][:] =-0.04
        W[att+"w0"][:] = 6*(n/(C-1))**(1+(i/(L-1))**0.3) - 6 + zigzag*2.5 + 0.5
        W[att+"a0"][:] =-0.19 + zigzag*0.3 + linear*0.4
        W[att+"v0"][:] = 0.73 - linear*0.4

        for c in "wvag":
            W[att+c+"1"][:] = 0
            nn.init.orthogonal_(W[att+c+"2"], gain=0.1)

        W[att+"ln_x.weight"][:] = ((1+i)/L)**0.7
        nn.init.orthogonal_(W[att+"receptance.weight"])
        nn.init.orthogonal_(W[att+"key.weight"], gain=0.1)
        nn.init.orthogonal_(W[att+"value.weight"])
        W[att+"output.weight"][:] = 0
    W = {k:v.bfloat16() for k,v in W.items()}
    return W


# Load dataset

class BinIdxDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args

        path = args.data_file
        with open(path+".idx", "rb") as stream:
            assert stream.read(9) == b"MMIDIDX\x00\x00" # File format magic
            assert struct.unpack("<Q", stream.read(8)) == (1,) # Version
            dtype_code, = struct.unpack("<B", stream.read(1))
            th = torch
            dtypes = [th.uint8, th.int8, th.int16, th.int32, th.int64, th.float, th.double, th.uint16]
            dtype = dtypes[dtype_code-1]

        data_size = os.path.getsize(path+".bin") // dtype.itemsize
        self.data = torch.from_file(path+".bin", dtype = dtype, size = data_size, shared = True)

        self.prime = len(self.data) // args.ctx_len - 1
        while self.prime%3 != 2 or any(self.prime%i == 0 for i in range(2,int(self.prime**0.5+1))):
            self.prime -= 1

    def __len__(self):
        return len(self.data) // (args.ctx_len * args.total_bsz)

    def __getitem__(self, idx):
        ctx_len = self.args.ctx_len

        i = 1 + idx * self.args.world_size + self.args.global_rank
        i = ((int(self.prime * (5**0.5-1)/2) * i**3) % self.prime) * ctx_len

        xy = self.data[i:i + ctx_len+1].to(torch.long)
        return xy[:-1], xy[1:]


# Encourage the logits to be close to 0
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, dloss):
        y = ctx.saved_tensors[0]
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.scatter(torch.zeros_like(y), -1, ids, maxx * factor)
        return dloss, gy


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    args = parse_args()

    if args.torch_compile: Block.forward = torch.compile(Block.forward)

    model = RWKV(args)

    # Sample initial model if needed
    if args.global_rank == 0:
        os.makedirs(args.proj_dir, exist_ok = True)
        filename = f"{args.proj_dir}/rwkv-init.pth"
        if not os.path.exists(filename):
            print(f'Saving initial model to {filename}')
            init_weights = sample_initial_weights(model, args)
            torch.save(init_weights, filename)

    # Find latest model and load it
    latest = max([int(p[5:-4]) for p in os.listdir(args.proj_dir) if p.startswith("rwkv-") and p.endswith(".pth") and p[5:-4].isdigit()], default=-1)
    start_epoch = latest+1
    if latest == -1: latest = 'init'
    filename = f"{args.proj_dir}/rwkv-{latest}.pth"

    deepspeed.init_distributed()
    deepspeed.comm.barrier() # Wait for initial model to be ready
    if args.global_rank == 0:
        print(f'Loading {filename}')
    model.load_state_dict(torch.load(filename))

    # Build optimizer
    lr_decay, lr_1x, lr_2x = [], [], []
    for n, p in model.named_parameters():
        if "att.w0" in n:
            lr_2x.append(p)
        elif p.ndim == 2 and ".weight" in n:
            lr_decay.append(p)
        else:
            lr_1x.append(p)
    assert lr_decay and lr_1x and lr_2x
    optim_groups = [ {"params": lr_1x, "lr_scale": 1}, {"params": lr_2x, "lr_scale": 2}, {"params": lr_decay, "lr_scale": 1} ]
    opt = deepspeed.ops.adam.FusedAdam(optim_groups, lr=-1, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    def lr_schedule(cur_step):
        dataset_steps = len(dataset.data) / (args.ctx_len * args.total_bsz)
        progress = (cur_step - args.warmup_steps) / (dataset_steps - args.warmup_steps)

        if progress < 0:
            return args.lr_init * (0.01 + 0.99 * cur_step / args.warmup_steps)
        else:
            decay = args.lr_final / args.lr_init
            return args.lr_init * 1/2*(1+decay + (1-decay)*math.cos(torch.pi * min(progress,1)))

    # Configure deepspeed
    ds_config = {
      "train_micro_batch_size_per_gpu": args.micro_bsz,
      "gradient_clipping": args.grad_clip,
      "bf16": { "enabled": True },
      "zero_optimization": {"stage": 2, 
                            "allgather_bucket_size": args.ds_bucket_mb<<20, 
                            "reduce_bucket_size": args.ds_bucket_mb<<20}}
    model, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, model_parameters=model.parameters(), config=ds_config)

    # Logging
    if args.global_rank == 0:
        log = open(f"{args.proj_dir}/train_log.txt", "a")
        print(f"NEW RUN {args.timestamp}\n{vars(args)}\n{model.config}", file=log, flush=True)
        if args.wandb:
            import wandb
            wandb.init(project=args.wandb, name=f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd} {args.timestamp}", config=args, save_code=False)
        last_time = time.time_ns()
        loss_cnt = loss_sum = 0

    # Training loop

    dataset = BinIdxDataset(args)
    epoch_steps = args.samples_per_epoch // args.total_bsz

    epochs = math.ceil(len(dataset.data) / (args.samples_per_epoch*args.ctx_len))
    for epoch in range(start_epoch, epochs):
        prog_bar = tqdm.tqdm(range(epoch_steps), desc=f'Epoch {epoch}', disable = args.global_rank>0)
        for local_step in prog_bar:
            cur_step = epoch_steps * epoch + local_step

            lr = lr_schedule(cur_step)
            for param_group in opt.param_groups:
                param_group["lr"] = lr * param_group["lr_scale"]
            opt.param_groups[2]["weight_decay"] = args.weight_decay

            # Load batch
            tokens, targets = map(torch.stack, zip(*[dataset[cur_step*args.micro_bsz+i] for i in range(args.micro_bsz)]))

            # Update step
            logits = model(tokens.cuda())
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.flatten().cuda())
            loss = L2Wrap.apply(loss, logits)
            model.backward(loss)
            model.step()

            all_loss = loss.detach().clone()/args.world_size
            torch.distributed.reduce(all_loss, 0, op=torch.distributed.ReduceOp.SUM)

            # Logging
            if args.global_rank == 0:
                tokens_per_step = args.ctx_len * args.total_bsz

                now = time.time_ns()
                it_s = 1e9/(now-last_time)
                kt_s = tokens_per_step * it_s / 1000
                last_time = now

                loss_sum += all_loss.item()
                loss_cnt += 1
                epoch_loss = loss_sum/loss_cnt

                info = {"loss": epoch_loss, "lr": lr, "last it/s": it_s, "Kt/s": kt_s}
                for k in info: info[k] = f'{info[k]:<5.4g}'.replace(' ','0')
                prog_bar.set_postfix(info)

                if args.wandb:
                    info = {"loss": loss, "lr": lr, "wd": args.weight_decay, "Gtokens": cur_step * tokens_per_step / 1e9, "kt/s": kt_s}
                    wandb.log(info, step=cur_step)

                # Save final model
                dataset_steps = len(dataset.data) / (args.ctx_len * args.total_bsz)
                if (cur_step+1)*tokens_per_step >= len(dataset.data):
                    torch.save(model.module.state_dict(), f"{args.proj_dir}/rwkv-final.pth")
                    exit(0)

        if args.global_rank == 0:
            # Save checkpoints
            if ((args.epoch_save and (epoch-start_epoch) % args.epoch_save == 0) or epoch == epochs-1):
                torch.save(model.module.state_dict(), f"{args.proj_dir}/rwkv-{epoch}.pth")

            # Logging
            print(f"{epoch} {epoch_loss:.6f} {math.exp(epoch_loss):.4f} {lr:.8f} {datetime.datetime.now()}", file=log, flush=True)
            loss_cnt = loss_sum = 0
