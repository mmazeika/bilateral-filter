#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>


#define divup(a, b) (((a)+(b)-1)/(b))

static const int THREADS_X = 16;
static const int THREADS_Y = 16;


template <typename scalar_t>
__global__ void bilateral_cuda_forward_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        scalar_t* __restrict__ numerator,
        scalar_t* __restrict__ denominator,
        const scalar_t* __restrict__ sigmas_v,
        const scalar_t* __restrict__ sigmas_s,
        const int window_size,
        const int num_shrd_elems,
        const int num_gauss_elems,
        const int blk_x,
        const int blk_y,
        const int batch_size,
        const int channels,
        const int side_len) {

    // declare shared memory
    extern __shared__ float shared[];
    float *local_mem = shared;
    float *gauss2d  = (float*)&local_mem[num_shrd_elems];

    // locate current thread in image
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int b = blockIdx.z / channels;  // batch index
    const int c = blockIdx.z % channels;  // channel index
    const int center_x = blockIdx.x * THREADS_X + lx;
    const int center_y = blockIdx.y * THREADS_Y + ly;

//    printf("b: %d, c: %d, center_x: %d, center_y: %d\n", b, c, center_x, center_y);

    // locate current thread in input (flat indexing)
    const int c_sizeof = side_len * side_len;
    const int b_sizeof = c_sizeof*channels;
    const int bc_offset = c*c_sizeof + b*b_sizeof;

    // other setup
    const int radius = window_size / 2;
    const int chunk_len_padded = THREADS_X + 2 * radius;
    const float sigma_v = sigmas_v[c];
    const float sigma_s = sigmas_s[c];
    const float inv_sigma_v = 1. / sigma_v;
    const float inv_sigma_s = 1. / sigma_s;
    const float inv_var_v_neg2 = -0.5 * inv_sigma_v * inv_sigma_v;
    const float inv_var_s_neg2 = -0.5 * inv_sigma_s * inv_sigma_s;
    const int center_idx = center_y + center_x*side_len + bc_offset;
    const int center_idx_shrd = (ly + radius) + (lx + radius) * chunk_len_padded;

    // fill gauss2d local mem (in parallel)
    if ((lx < window_size) && (ly < window_size)) {
        const float tmp1 = (lx - radius);
        const float tmp2 = (ly - radius);
        gauss2d[ly + lx*window_size] = __expf(((tmp1 * tmp1) + (tmp2 * tmp2)) * inv_var_s_neg2);
    }

    // pull image chunk to local memory (in parallel) (for loops are just to get the padding)
    for (int i = lx; i < chunk_len_padded; i += THREADS_X) {
        for (int j = ly; j < (THREADS_Y + 2 * radius); j += THREADS_Y) {
            const int load_x = blockIdx.x * THREADS_X + i - radius;
            const int load_y = blockIdx.y * THREADS_Y + j - radius;
//            printf("load_x: %d, load_y: %d\n", load_x, load_y);
            if ((load_x >= 0) && (load_x < side_len) && (load_y >= 0) && (load_y < side_len)) {
                local_mem[i * chunk_len_padded + j] = input[load_y + load_x*side_len + bc_offset];
//                printf("normal case:: load_x: %d, load_y: %d\n", load_x, load_y);
            } else {
                local_mem[i * chunk_len_padded + j] = 0;
//                printf("edge case:: load_x: %d, load_y: %d\n", load_x, load_y);
            }
        }
    }

    __syncthreads();

    // main bilateral filtering code
    if ((center_x < side_len) && (center_y < side_len)) {
        float res  = 0.;
        float norm = 0.;
        const float center_val = local_mem[center_idx_shrd];
#pragma unroll
        for (int xi = 0; xi < window_size; xi++) {
            const int xi_in = center_x + (xi - radius);
#pragma unroll
            for (int yi = 0; yi < window_size; yi++) {
                const int yi_in = center_y + (yi - radius);
//                printf("xi_in: %d, yi_in: %d\n", xi_in, yi_in);

                if ((xi_in >= 0) && (xi_in < side_len) && (yi_in >= 0) && (yi_in < side_len)) {
                    const int offset_idx_shrd = (ly + yi) + (lx + xi) * chunk_len_padded;
//                    const int offset_idx = center_idx + (yi - radius) + (xi - radius) * side_len;
                    const float offset_val = local_mem[offset_idx_shrd];
                    const float diff = center_val - offset_val;
                    const float weight = gauss2d[yi + xi*window_size] * __expf(diff * diff * inv_var_v_neg2);
                    res  += offset_val * weight;
                    norm += weight;
                }
            }
        }
        output[center_idx] = res / norm;
        numerator[center_idx] = res;
        denominator[center_idx] = norm;
    }
}


template <typename scalar_t>
__global__ void bilateral_cuda_backward_kernel(
        const scalar_t* __restrict__ grad_output,
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_sigma_v,
        scalar_t* __restrict__ grad_sigma_s,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ sigmas_v,
        const scalar_t* __restrict__ sigmas_s,
        scalar_t* __restrict__ numerator,
        scalar_t* __restrict__ denominator,
        const int window_size,
        const int num_shrd_elems,
        const int num_gauss_elems,
        const int blk_x,
        const int blk_y,
        const int batch_size,
        const int channels,
        const int side_len) {

    // declare shared memory
    extern __shared__ float shared[];
    float *local_mem_input = shared;
//    float *local_mem_grad_output = (float*)&local_mem_input[num_shrd_elems];
//    float *local_mem_numerator = (float*)&local_mem_input[num_shrd_elems];
//    float *local_mem_denominator = (float*)&local_mem_numerator[num_shrd_elems];
    float *gauss2d  = (float*)&local_mem_input[num_shrd_elems];

    // locate current thread in image
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;
    const int b = blockIdx.z / channels;  // batch index
    const int c = blockIdx.z % channels;  // channel index
    const int center_x = blockIdx.x * THREADS_X + lx;
    const int center_y = blockIdx.y * THREADS_Y + ly;

//    printf("b: %d, c: %d, center_x: %d, center_y: %d\n", b, c, center_x, center_y);

    // locate current thread in input (flat indexing)
    const int c_sizeof = side_len * side_len;
    const int b_sizeof = c_sizeof*channels;
    const int bc_offset = c*c_sizeof + b*b_sizeof;

    // other setup
    const int radius = window_size / 2;
    const int chunk_len_padded = THREADS_X + 2 * radius;
    const float sigma_v = sigmas_v[c];
    const float sigma_s = sigmas_s[c];
    const float inv_sigma_v = 1. / sigma_v;
    const float inv_sigma_s = 1. / sigma_s;
    const float inv_var_v = inv_sigma_v * inv_sigma_v;
//    const float var_s = sigma_s * sigma_s;
    const float inv_var_v_neg2 = -0.5 * inv_sigma_v * inv_sigma_v;
    const float inv_var_s_neg2 = -0.5 * inv_sigma_s * inv_sigma_s;
    const int center_idx = center_y + center_x*side_len + bc_offset;
    const int center_idx_shrd = (ly + radius) + (lx + radius) * chunk_len_padded;

    // fill gauss2d local mem (in parallel)
    if ((lx < window_size) && (ly < window_size)) {
        const float tmp1 = (lx - radius);
        const float tmp2 = (ly - radius);
        gauss2d[ly + lx*window_size] = __expf(((tmp1 * tmp1) + (tmp2 * tmp2)) * inv_var_s_neg2);
    }

    // pull chunks to local memory (in parallel) (for loops are just to get the padding)
    for (int i = lx; i < chunk_len_padded; i += THREADS_X) {
        for (int j = ly; j < (THREADS_Y + 2 * radius); j += THREADS_Y) {
            const int load_x = blockIdx.x * THREADS_X + i - radius;
            const int load_y = blockIdx.y * THREADS_Y + j - radius;
            const int tmp_idx = i * chunk_len_padded + j;
            const int tmp1_idx = load_y + load_x*side_len + bc_offset;
            if ((load_x >= 0) && (load_x < side_len) && (load_y >= 0) && (load_y < side_len)) {
                local_mem_input[tmp_idx] = input[tmp1_idx];
//                local_mem_grad_output[tmp_idx] = grad_output[tmp1_idx];
//                local_mem_numerator[tmp_idx] = numerator[tmp1_idx];
//                local_mem_denominator[tmp_idx] = denominator[tmp1_idx];
            } else {
                local_mem_input[tmp_idx] = 0.;
//                local_mem_grad_output[tmp_idx] = 0;
//                local_mem_numerator[tmp_idx] = 0;
//                local_mem_denominator[tmp_idx] = 0;
            }
        }
    }

    __syncthreads();

    // main loop (if statement just in case)
    if ((center_x < side_len) && (center_y < side_len)) {
        float dN_input_center = 0.;
        float dD_input_center = 0.;
        float grad_input_accum = 0.;
        float dN_sigma_v = 0.;
        float dD_sigma_v = 0.;
        float dN_sigma_s = 0.;
        float dD_sigma_s = 0.;

        const float center_val = local_mem_input[center_idx_shrd];
#pragma unroll
        for (int xi = 0; xi < window_size; xi++) {
            const int xi_in = center_x + (xi - radius);
#pragma unroll
            for (int yi = 0; yi < window_size; yi++) {
                const int yi_in = center_y + (yi - radius);

                if ((xi_in >= 0) && (xi_in < side_len) && (yi_in >= 0) && (yi_in < side_len)) {
                    const int offset_idx = center_idx + (yi - radius) + (xi - radius) * side_len;
                    const int offset_idx_shrd = (ly + yi) + (lx + xi) * chunk_len_padded;
                    const float offset_val = local_mem_input[offset_idx_shrd];
                    const float diff = center_val - offset_val;
                    const float weight = gauss2d[yi + xi*window_size] * __expf(diff * diff * inv_var_v_neg2);

                    if (((xi - radius) == 0) && ((yi - radius) == 0)) {
                        dN_input_center += 1;
                    } else {
                        dN_input_center += weight * offset_val * (-diff * inv_var_v);
                        dD_input_center += weight * (-diff * inv_var_v);

                        float tmp = weight * diff * diff;
                        dN_sigma_v += tmp * offset_val;
                        dD_sigma_v += tmp;

                        tmp = weight * ((xi - radius) * (xi - radius) + (yi - radius) * (yi - radius));
                        dN_sigma_s += tmp * offset_val;
                        dD_sigma_s += tmp;

                        const float dN_gi = weight * ((-diff * inv_var_v) * center_val + 1);  // -diff for a completely different reason than above
                        const float dD_gi = weight * (-diff * inv_var_v);
                        const float numer = numerator[offset_idx];
                        const float denom = denominator[offset_idx];
                        const float d_out_d_in = (dN_gi * denom - numer * dD_gi) / (denom * denom);
                        grad_input_accum += grad_output[offset_idx] * d_out_d_in;
                    }
                }
            }
        }
        const float numer = numerator[center_idx];
        const float denom = denominator[center_idx];
        const float inv_denom_sq = 1. / (denom * denom);
        float d_out_d_in = (dN_input_center * denom - numer * dD_input_center) * inv_denom_sq;
        const float grad_out = grad_output[center_idx];
        grad_input_accum += grad_out * d_out_d_in;
        grad_input[center_idx] = grad_input_accum;

        float tmp = inv_sigma_v * inv_sigma_v * inv_sigma_v;
        dN_sigma_v *= tmp;
        dD_sigma_v *= tmp;
        d_out_d_in = (dN_sigma_v * denom - numer * dD_sigma_v) * inv_denom_sq;
        grad_sigma_v[c] += grad_out * d_out_d_in;
//        printf("addition to grad_sigma_v: %f, dN_sigma_v: %f, denom: %f, numer: %f, dD_sigma_v %f\n",
//               grad_out * d_out_d_in, dN_sigma_v, denom, numer, dD_sigma_v);

        tmp = inv_sigma_s * inv_sigma_s * inv_sigma_s;
        dN_sigma_s *= tmp;
        dD_sigma_s *= tmp;
        d_out_d_in = (dN_sigma_s * denom - numer * dD_sigma_s) * inv_denom_sq;
        grad_sigma_s[c] += grad_out * d_out_d_in;
//        printf("addition to grad_sigma_s: %f\n", grad_out * d_out_d_in);
    }
}


std::vector<at::Tensor> bilateral_cuda_forward(at::Tensor input, at::Tensor sigma_v, at::Tensor sigma_s) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int side_len = input.size(2);
    const int window_size = 5;
    const int radius = window_size / 2;


    dim3 threads(THREADS_X, THREADS_Y);

    const int blk_x = divup(side_len, THREADS_X);
    const int blk_y = divup(side_len, THREADS_Y);

    dim3 blocks(blk_x, blk_y, batch_size*channels);


    const int num_shrd_elems    = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    const int num_gauss_elems   = window_size * window_size;
    size_t total_shrd_size   = sizeof(float) * (num_shrd_elems + num_gauss_elems);

    if (batch_size*channels > 65535) {
        printf("\nCUDA bilateral filter doesn't support batch size %d with %d channels\n", batch_size, channels);
        exit(1);
    }

    auto output = at::zeros_like(input);
    auto numerator = at::zeros_like(input);  // for backprop
    auto denominator = at::zeros_like(input);  // for backprop

    AT_DISPATCH_FLOATING_TYPES(input.type(), "bilateral_forward_cuda", ([&] {
        bilateral_cuda_forward_kernel<scalar_t><<<blocks, threads, total_shrd_size>>>(
                input.data<scalar_t>(), output.data<scalar_t>(), numerator.data<scalar_t>(), denominator.data<scalar_t>(),
                        sigma_v.data<scalar_t>(), sigma_s.data<scalar_t>(), window_size, num_shrd_elems,
                        num_gauss_elems, blk_x, blk_y, batch_size, channels, side_len);
    }));

    cudaDeviceSynchronize();

    return {output, numerator, denominator};
}


std::vector<at::Tensor> bilateral_cuda_backward(at::Tensor grad_output, at::Tensor input, at::Tensor sigma_v,
                                                at::Tensor sigma_s, at::Tensor numerator, at::Tensor denominator) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int side_len = input.size(2);
    const int window_size = 5;
    const int radius = window_size / 2;


    dim3 threads(THREADS_X, THREADS_Y);

    const int blk_x = divup(side_len, THREADS_X);
    const int blk_y = divup(side_len, THREADS_Y);

    dim3 blocks(blk_x, blk_y, batch_size*channels);


    const int num_shrd_elems    = (THREADS_X + 2 * radius) * (THREADS_Y + 2 * radius);
    const int num_gauss_elems   = window_size * window_size;
    // 4 * num_shrd_elems for input, grad_output, numerator, denominator
    size_t total_shrd_size   = sizeof(float) * (1 * num_shrd_elems + num_gauss_elems);

    if (total_shrd_size > 49151) {
        printf("\nShared memory not large enough\n");
        exit(1);
    }

    if (batch_size*channels > 65535) {
        printf("\nCUDA bilateral filter doesn't support batch size %d with %d channels\n", batch_size, channels);
        exit(1);
    }

    auto grad_input = at::zeros_like(input);
    auto grad_sigma_v = at::zeros_like(sigma_v);
    auto grad_sigma_s = at::zeros_like(sigma_s);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "bilateral_backward_cuda", ([&] {
        bilateral_cuda_backward_kernel<scalar_t><<<blocks, threads, total_shrd_size>>>(
                grad_output.data<scalar_t>(), grad_input.data<scalar_t>(), grad_sigma_v.data<scalar_t>(),
                        grad_sigma_s.data<scalar_t>(), input.data<scalar_t>(), sigma_v.data<scalar_t>(),
                        sigma_s.data<scalar_t>(), numerator.data<scalar_t>(), denominator.data<scalar_t>(), window_size,
                        num_shrd_elems, num_gauss_elems, blk_x, blk_y, batch_size, channels, side_len);
    }));

    cudaDeviceSynchronize();

    return {grad_input, grad_sigma_v, grad_sigma_s};
}