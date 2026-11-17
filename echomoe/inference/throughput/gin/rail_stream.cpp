// Python binding for rail_stream.cu.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

int64_t stream_create(int64_t comm, int max_t, int max_width, int k, int experts, int contexts);
void stream_dispatch(int64_t stream, const void* x, const float* xs, const int64_t* topk_idx,
                     const float* topk_weights, int n, int vbytes, int sbytes, int t, int ctas, int64_t chunk,
                     void* recv_x, float* recv_xs, int64_t* recv_idx, float* recv_weights, int* recv_src, int* psum,
                     int* comb_rows, int* node_rows, cudaStream_t cs);
void stream_combine(int64_t stream, const void* y, int* psum, int* comb_rows, int* node_rows, int n, int width, int t,
                    int ctas, int64_t chunk, bool fp8, void* out, cudaStream_t cs);
void stream_destroy(int64_t stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &stream_create);
    // x: [n, vbytes] of any element type; xs: [n, sbytes / 4] float32 scales, or empty
    m.def("dispatch", [](int64_t stream, torch::Tensor x, torch::Tensor xs, torch::Tensor topk_idx,
                         torch::Tensor topk_weights, int t, int ctas, int64_t chunk, torch::Tensor recv_x,
                         torch::Tensor recv_xs, torch::Tensor recv_idx, torch::Tensor recv_weights,
                         torch::Tensor recv_src, torch::Tensor psum, torch::Tensor comb_rows,
                         torch::Tensor node_rows) {
        const bool scaled = xs.dim() == 2 && xs.size(1) > 0;
        stream_dispatch(stream, x.data_ptr(), scaled ? xs.data_ptr<float>() : nullptr, topk_idx.data_ptr<int64_t>(),
                        topk_weights.data_ptr<float>(), x.size(0), int(x.size(1) * x.element_size()),
                        scaled ? int(xs.size(1)) * 4 : 0, t, ctas, chunk, recv_x.data_ptr(),
                        scaled ? recv_xs.data_ptr<float>() : nullptr, recv_idx.data_ptr<int64_t>(),
                        recv_weights.data_ptr<float>(), recv_src.data_ptr<int>(), psum.data_ptr<int>(),
                        comb_rows.data_ptr<int>(), node_rows.data_ptr<int>(), c10::cuda::getCurrentCUDAStream());
    });
    m.def("combine", [](int64_t stream, torch::Tensor y, torch::Tensor psum, torch::Tensor comb_rows,
                        torch::Tensor node_rows, int n, int t, int ctas, int64_t chunk, bool fp8,
                        torch::Tensor out) {
        stream_combine(stream, y.data_ptr<at::BFloat16>(), psum.data_ptr<int>(), comb_rows.data_ptr<int>(),
                       node_rows.data_ptr<int>(), n, y.size(1), t, ctas, chunk, fp8, out.data_ptr<at::BFloat16>(),
                       c10::cuda::getCurrentCUDAStream());
    });
    m.def("destroy", &stream_destroy);
}
