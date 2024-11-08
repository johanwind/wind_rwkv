import torch
import torch.nn as nn
import torch.nn.functional as F
import wind_rwkv

class TimeMixer(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_embd = args.n_embd
        args.dim_att = args.n_embd

        self.n_head = args.n_head
        self.head_size = args.dim_att // args.n_head
        assert args.dim_att % self.n_head == 0

        wind_rwkv.load_attn_kernel(self.head_size)

        ratio_0_to_1 = layer_id / max(args.n_layer - 1, 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, args.n_embd)
        for i in range(args.n_embd):
            ddd[0, 0, i] = i / args.n_embd

        # initialization comes from fitting my RWKV-6 7B runs
        # merging r&g w&a to save params
        self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
        time_maa_rg = 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
        time_maa_wa = 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
        time_maa_k = 1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
        time_maa_v = 1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
        self.time_maa = nn.Parameter(torch.stack([time_maa_rg, time_maa_wa, time_maa_k, time_maa_v]))

        decay_speed = torch.ones(args.dim_att)
        for n in range(args.dim_att):
            decay_speed[n] = -7 + 5 * (n / (args.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
        self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

        self.time_faaaa = nn.Parameter(torch.zeros(args.dim_att))
        self.time_aaaaa = nn.Parameter(torch.zeros(1,1,args.dim_att))

        def ortho_init(x, scale):
            shape = x.shape
            assert len(shape) in [2,3]
            if len(shape) == 2:
                gain = (shape[0] / shape[1])**0.5 if shape[0] > shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
            elif len(shape) == 3:
                gain = (shape[1] / shape[2])**0.5 if shape[1] > shape[2] else 1
                for i in range(shape[0]):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            return x

        D_MIX_LORA = 32
        self.time_maa_w1 = nn.Parameter(torch.zeros(D_MIX_LORA*4, args.n_embd))
        self.time_maa_w2 = nn.Parameter(ortho_init(torch.zeros(4, D_MIX_LORA, args.n_embd), 0.1).mT.contiguous())

        D_DECAY_LORA = 64
        self.time_decay_w1 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.n_embd))
        self.time_decay_w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, args.dim_att), 0.1).mT.contiguous())

        D_AAA_LORA = 16
        self.time_aaa_w1 = nn.Parameter(torch.zeros(D_AAA_LORA, args.n_embd))
        self.time_aaa_w2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, args.dim_att), 0.1).mT.contiguous())

        D_KKK_LORA = 16
        self.time_kkk_w1 = nn.Parameter(torch.zeros(D_KKK_LORA, args.n_embd))
        self.time_kkk_w2 = nn.Parameter(ortho_init(torch.zeros(D_KKK_LORA, args.dim_att), 0.1).mT.contiguous())

        D_GATE_LORA = 128
        self.gate_w1 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, args.n_embd), 0.1))
        self.gate_w2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, args.dim_att), 0.1).mT.contiguous())

        D_MA_LORA = 16
        self.ma_w1 = nn.Parameter(torch.zeros(D_MA_LORA, args.n_embd))
        self.ma_w2 = nn.Parameter(ortho_init(torch.zeros(D_MA_LORA, args.dim_att), 0.1).mT.contiguous())
        self.time_misc_a = nn.Parameter(torch.zeros(1,1,args.n_embd))
        D_MK_LORA = 16
        self.mk_w1 = nn.Parameter(torch.zeros(D_MK_LORA, args.n_embd))
        self.mk_w2 = nn.Parameter(ortho_init(torch.zeros(D_MK_LORA, args.dim_att), 0.1).mT.contiguous())
        self.time_misc_k = nn.Parameter(torch.zeros(1,1,args.n_embd))

        def uniform(m, n, scale):
            return nn.Parameter(torch.empty(n, m).uniform_(-scale/m**0.5, scale/m**0.5))
        self.receptance = uniform(args.n_embd, args.dim_att, 0.5)
        self.key = uniform(args.n_embd, args.dim_att, 0.05)
        self.value = uniform(args.n_embd, args.dim_att, 0.5)
        self.output = uniform(args.dim_att, args.n_embd, 0)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=64e-5)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        xrg,xwa,xk,xv = wind_rwkv.fused_time_shift(x.bfloat16(), self.time_maa_x, self.time_maa_w1, self.time_maa_w2, self.time_maa)

        r = xrg @ self.receptance.mT
        g1 = xrg @ self.gate_w1.mT

        k = xk @ self.key.mT
        kk1,mk1 = (xk @ torch.cat([self.time_kkk_w1,self.mk_w1]).mT).split([16,16],dim=2)

        v = xv @ self.value.mT

        w1,a1,ma1 = (xwa @ torch.cat([self.time_decay_w1,self.time_aaa_w1,self.ma_w1]).mT).split([64,16,16],dim=2)

        g = torch.tanh(g1) @ self.gate_w2.mT
        w = -F.softplus(-(self.time_decay + torch.tanh(w1) @ self.time_decay_w2.mT)) - 0.5

        kk = k + torch.tanh(kk1) @ self.time_kkk_w2.mT
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C).bfloat16()
        a = torch.sigmoid( self.time_aaaaa + a1 @ self.time_aaa_w2.mT )

        ma = torch.sigmoid(self.time_misc_a + ma1 @ self.ma_w2.mT)
        k = k * ma + k*a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + mk1 @ self.mk_w2.mT)
        k = k * (w*mk).exp()
        b = -kk*a

        x = wind_rwkv.fused_attn_ln(r.bfloat16(), w.bfloat16(), k.bfloat16(), v.bfloat16(), kk.bfloat16(), b.bfloat16(), g.bfloat16(), torch.stack([self.ln_x.weight, self.ln_x.bias, self.time_faaaa]), C//H)
        return x @ self.output.mT
