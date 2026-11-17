// Python binding for rail_moe.cu.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

int64_t rail_create(int64_t comm, int max_t, int max_width, int k, int experts, int contexts, int gather_bytes,
                    int reduce_width, bool fp8);
void rail_dispatch(int64_t rail, const void* x, const float* xs, const int64_t* topk_idx, const float* topk_weights,
                   int n, int vbytes, int sbytes, int t, int ctas, void* recv_x, float* recv_xs, int64_t* recv_idx,
                   float* recv_weights, int* recv_src, int* psum, int* comb_rows, int* node_rows,
                   cudaStream_t stream);
void rail_combine(int64_t rail, const void* y, int* psum, int* comb_rows, int* node_rows, int n, int width, int t,
                  int ctas, void* out, cudaStream_t stream);
void rail_all_gather(int64_t rail, const void* in, void* out, int t, int64_t row, int ctas, cudaStream_t stream);
void rail_reduce_scatter(int64_t rail, const void* in, void* out, int t, int width, int ctas, bool fp8,
                         cudaStream_t stream);
void rail_destroy(int64_t rail);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &rail_create);
    // x: [n, vbytes] of any element type; xs: [n, sbytes / 4] float32 scales, or empty
    m.def("dispatch", [](int64_t rail, torch::Tensor x, torch::Tensor xs, torch::Tensor topk_idx,
                         torch::Tensor topk_weights, int t, int ctas, torch::Tensor recv_x, torch::Tensor recv_xs,
                         torch::Tensor recv_idx, torch::Tensor recv_weights, torch::Tensor recv_src, torch::Tensor psum,
                         torch::Tensor comb_rows, torch::Tensor node_rows) {
        const bool scaled = xs.dim() == 2 && xs.size(1) > 0;
        rail_dispatch(rail, x.data_ptr(), scaled ? xs.data_ptr<float>() : nullptr, topk_idx.data_ptr<int64_t>(),
                      topk_weights.data_ptr<float>(), x.size(0), int(x.size(1) * x.element_size()),
                      scaled ? int(xs.size(1)) * 4 : 0, t, ctas, recv_x.data_ptr(),
                      scaled ? recv_xs.data_ptr<float>() : nullptr, recv_idx.data_ptr<int64_t>(),
                      recv_weights.data_ptr<float>(), recv_src.data_ptr<int>(), psum.data_ptr<int>(),
                      comb_rows.data_ptr<int>(), node_rows.data_ptr<int>(), c10::cuda::getCurrentCUDAStream());
    });
    m.def("combine", [](int64_t rail, torch::Tensor y, torch::Tensor psum, torch::Tensor comb_rows,
                        torch::Tensor node_rows, int n, int t, int ctas, torch::Tensor out) {
        rail_combine(rail, y.data_ptr<at::BFloat16>(), psum.data_ptr<int>(), comb_rows.data_ptr<int>(),
                     node_rows.data_ptr<int>(), n, y.size(1), t, ctas, out.data_ptr<at::BFloat16>(),
                     c10::cuda::getCurrentCUDAStream());
    });
    m.def("all_gather", [](int64_t rail, torch::Tensor in, torch::Tensor out, int ctas) {
        rail_all_gather(rail, in.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), in.size(0), in.size(1), ctas,
                        c10::cuda::getCurrentCUDAStream());
    });
    m.def("reduce_scatter", [](int64_t rail, torch::Tensor in, torch::Tensor out, int ctas, bool fp8) {
        rail_reduce_scatter(rail, in.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(), out.size(0), out.size(1),
                            ctas, fp8, c10::cuda::getCurrentCUDAStream());
    });
    m.def("destroy", &rail_destroy);
}
