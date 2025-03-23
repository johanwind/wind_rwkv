// Copyright (c) 2024, Johan Sokrates Wind

#include <torch/extension.h>
#include <cuda_bf16.h>
using bf = __nv_bfloat16;
using torch::Tensor;

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*s0, bf*y, bf*s, bf*sT);

void forward(Tensor &w, Tensor &q, Tensor &k, Tensor &v, Tensor &a, Tensor &b, Tensor &s0, Tensor &y, c10::optional<torch::Tensor> s, Tensor &sT) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)s0.data_ptr(), (bf*)y.data_ptr(), s.has_value() ? (bf*)s.value().data_ptr() : NULL, (bf*)sT.data_ptr());
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*a, bf*b, bf*dy, bf*s, bf*dsT, bf*dw, bf*dq, bf*dk, bf*dv, bf*da, bf*db, bf*ds0);

void backward(Tensor &w, Tensor &q, Tensor &k, Tensor &v, Tensor &a, Tensor &b, Tensor &dy,
        Tensor &s, Tensor &dsT, Tensor &dw, Tensor &dq, Tensor &dk, Tensor &dv, Tensor &da, Tensor &db, Tensor &ds0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(), 
            (bf*)s.data_ptr(), (bf*)dsT.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr(), (bf*)ds0.data_ptr());
}

TORCH_LIBRARY(wind_chunked_cuda, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor s0, Tensor(a!) y, Tensor? s, Tensor(c!) sT) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor dsT, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db, Tensor(g!) ds0) -> ()");
}

TORCH_LIBRARY_IMPL(wind_chunked_cuda, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
