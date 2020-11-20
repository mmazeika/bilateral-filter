#include <torch/torch.h>
#include <vector>
#include <pybind11/pybind11.h>

// CUDA forward declarations

std::vector<at::Tensor> bilateral_cuda_forward(at::Tensor input, at::Tensor sigma_v, at::Tensor sigma_s);

std::vector<at::Tensor> bilateral_cuda_backward(at::Tensor grad_output, at::Tensor input, at::Tensor sigma_v,
                                                at::Tensor sigma_s, at::Tensor numerator, at::Tensor denominator);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> bilateral_forward(at::Tensor input, at::Tensor sigma_v, at::Tensor sigma_s) {
    CHECK_INPUT(input);
    CHECK_INPUT(sigma_v);
    CHECK_INPUT(sigma_s);

    if ((sigma_v.size(0) != input.size(1)) || (sigma_s.size(0) != input.size(1))) {
        printf("mismatched sigmas and input channels");
        exit(1);
    }

    return bilateral_cuda_forward(input, sigma_v, sigma_s);
}


std::vector<at::Tensor> bilateral_backward(at::Tensor grad_output, at::Tensor input, at::Tensor sigma_v,
                                           at::Tensor sigma_s, at::Tensor numerator, at::Tensor denominator) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(sigma_v);
    CHECK_INPUT(sigma_s);
    CHECK_INPUT(numerator);
    CHECK_INPUT(denominator);

    if ((sigma_v.size(0) != input.size(1)) || (sigma_s.size(0) != input.size(1))) {
        printf("mismatched sigmas and input channels");
        exit(1);
    }

    return bilateral_cuda_backward(grad_output, input, sigma_v, sigma_s, numerator, denominator);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &bilateral_forward, "Bilateral Forward", py::arg("input"), py::arg("sigma_v"), py::arg("sigma_s"));
m.def("backward", &bilateral_backward, "Bilateral Backward", py::arg("grad_output"), py::arg("input"),
py::arg("sigma_v"), py::arg("sigma_s"), py::arg("numerator"), py::arg("denominator"));
}