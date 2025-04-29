// Copyright (c) 2024, Johan Sokrates Wind

#include <torch/extension.h>

struct __nv_bfloat16;
using bf = __nv_bfloat16;
using torch::Tensor;

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*s0, bf*y, bf*s, bf*sT, long long*cu_seqlens);

void forward(Tensor &w, Tensor &q, Tensor &k, Tensor &v, Tensor &a, Tensor &b, Tensor &s0, Tensor &y, c10::optional<torch::Tensor> s, Tensor &sT, Tensor &cu_seqlens) {
    int B = cu_seqlens.sizes()[0]-1, T = w.sizes()[0], H = w.sizes()[1];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)s0.data_ptr(), (bf*)y.data_ptr(), s.has_value() ? (bf*)s.value().data_ptr() : NULL, (bf*)sT.data_ptr(), (long long*)cu_seqlens.data_ptr());
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*dy, bf*s, bf*dsT, bf*dw, bf*dq, bf*dk, bf*dv, bf*da, bf*db, bf*ds0, long long*cu_seqlens);

void backward(Tensor &w, Tensor &q, Tensor &k, Tensor &v, Tensor &a, Tensor &b, Tensor &dy,
        Tensor &s, Tensor &dsT, Tensor &dw, Tensor &dq, Tensor &dk, Tensor &dv, Tensor &da, Tensor &db, Tensor &ds0, Tensor &cu_seqlens) {
    int B = cu_seqlens.sizes()[0]-1, T = w.sizes()[0], H = w.sizes()[1];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(), 
            (bf*)s.data_ptr(), (bf*)dsT.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr(), (bf*)ds0.data_ptr(), (long long*)cu_seqlens.data_ptr());
}

TORCH_LIBRARY(wind_chunked_cuda_varlen, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor s0, Tensor(a!) y, Tensor? s, Tensor(c!) sT, Tensor cu_seqlens) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor dsT, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db, Tensor(g!) ds0, Tensor cu_seqlens) -> ()");
}

TORCH_LIBRARY_IMPL(wind_chunked_cuda_varlen, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
