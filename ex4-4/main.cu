/*
Now, with a single input array calculate and return the results on a different array:

    The exponentiation of the array elements to the power of 2
    The square root of the elements of the array
*/

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void array_square(const float *d_input, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = d_input[i] * d_input[i];
    }
}

__global__ void array_square_root(const float *d_input, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = sqrtf(d_input[i]);
    }
}

#ifndef UNIT_TEST
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

static void print_result(const char *label, const float *values, int count) {
    printf("%s:", label);
    for (int i = 0; i < count; ++i) {
        printf(" %.3f", values[i]);
    }
    printf("\n");
}

int main() {
    const int count = 6;
    const float h_input[] = {0.0f, 1.0f, 2.25f, 4.0f, 9.0f, 16.0f};
    float h_square[count];
    float h_square_root[count];
    float *d_input = NULL;
    float *d_square = NULL;
    float *d_square_root = NULL;
    const size_t bytes = count * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_square, bytes));
    CUDA_CHECK(cudaMalloc(&d_square_root, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    array_square<<<num_blocks, block_size>>>(d_input, d_square, count);
    array_square_root<<<num_blocks, block_size>>>(d_input, d_square_root, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_square, d_square, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_square_root, d_square_root, bytes, cudaMemcpyDeviceToHost));

    print_result("input", h_input, count);
    print_result("square", h_square, count);
    print_result("square root", h_square_root, count);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_square));
    CUDA_CHECK(cudaFree(d_square_root));

    return 0;
}
#endif
