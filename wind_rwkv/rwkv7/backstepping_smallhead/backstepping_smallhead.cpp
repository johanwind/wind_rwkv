#include <torch/extension.h>

struct __nv_bfloat16;
using bf = __nv_bfloat16;

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, float*s0, bf*y, float*s, float*sa, float*sT);

void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &s0, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa, torch::Tensor &sT) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (float*)s0.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)sT.data_ptr());
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, float*dsT, bf*dw, bf*dq, bf*dk, bf*dv, bf*da, bf*db, float*ds0);

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dsT, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db, torch::Tensor &ds0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)dsT.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr(), (float*)ds0.data_ptr());
}

TORCH_LIBRARY(wind_backstepping_smallhead, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor s0, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa, Tensor(d!) sT) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor dsT, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db, Tensor(g!) ds0) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping_smallhead, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
