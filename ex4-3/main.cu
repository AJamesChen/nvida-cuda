/*
Now, with two input arrays calculate the element-wise (corresponding elements on each index):

    Vector multiplication
    Vector division
    Vector absolute difference
    Vector maximum - the result array should receive the maximum of the two input elements
    Vector minimum - the result array should receive the minimum of the two input elements
    The modulus of the element from the first array and the second array
*/

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_multiply(const float *d_input_1, const float *d_input_2, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = d_input_1[i] * d_input_2[i];
    }
}

__global__ void vector_divide(const float *d_input_1, const float *d_input_2, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = d_input_1[i] / d_input_2[i];
    }
}

__global__ void vector_absolute_difference(const float *d_input_1, const float *d_input_2, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = fabsf(d_input_1[i] - d_input_2[i]);
    }
}

__global__ void vector_maximum(const float *d_input_1, const float *d_input_2, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = fmaxf(d_input_1[i], d_input_2[i]);
    }
}

__global__ void vector_minimum(const float *d_input_1, const float *d_input_2, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = fminf(d_input_1[i], d_input_2[i]);
    }
}

__global__ void vector_modulus(const float *d_input_1, const float *d_input_2, float *d_output, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = fmodf(d_input_1[i], d_input_2[i]);
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
    const float h_input1[] = {12.0f, 9.0f, -4.0f, 7.5f, 0.0f, 25.0f};
    const float h_input2[] = {5.0f, 3.0f, 2.0f, -2.5f, 4.0f, 6.0f};
    float h_multiply[count];
    float h_divide[count];
    float h_abs_diff[count];
    float h_maximum[count];
    float h_minimum[count];
    float h_modulus[count];
    float *d_input1 = NULL;
    float *d_input2 = NULL;
    float *d_multiply = NULL;
    float *d_divide = NULL;
    float *d_abs_diff = NULL;
    float *d_maximum = NULL;
    float *d_minimum = NULL;
    float *d_modulus = NULL;
    const size_t bytes = count * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input1, bytes));
    CUDA_CHECK(cudaMalloc(&d_input2, bytes));
    CUDA_CHECK(cudaMalloc(&d_multiply, bytes));
    CUDA_CHECK(cudaMalloc(&d_divide, bytes));
    CUDA_CHECK(cudaMalloc(&d_abs_diff, bytes));
    CUDA_CHECK(cudaMalloc(&d_maximum, bytes));
    CUDA_CHECK(cudaMalloc(&d_minimum, bytes));
    CUDA_CHECK(cudaMalloc(&d_modulus, bytes));

    CUDA_CHECK(cudaMemcpy(d_input1, h_input1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input2, h_input2, bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    vector_multiply<<<num_blocks, block_size>>>(d_input1, d_input2, d_multiply, count);
    vector_divide<<<num_blocks, block_size>>>(d_input1, d_input2, d_divide, count);
    vector_absolute_difference<<<num_blocks, block_size>>>(d_input1, d_input2, d_abs_diff, count);
    vector_maximum<<<num_blocks, block_size>>>(d_input1, d_input2, d_maximum, count);
    vector_minimum<<<num_blocks, block_size>>>(d_input1, d_input2, d_minimum, count);
    vector_modulus<<<num_blocks, block_size>>>(d_input1, d_input2, d_modulus, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_multiply, d_multiply, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_divide, d_divide, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_abs_diff, d_abs_diff, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_maximum, d_maximum, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_minimum, d_minimum, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_modulus, d_modulus, bytes, cudaMemcpyDeviceToHost));

    print_result("input1", h_input1, count);
    print_result("input2", h_input2, count);
    printf("----------------------------\n");
    print_result("multiply", h_multiply, count);
    print_result("divide", h_divide, count);
    print_result("absolute difference", h_abs_diff, count);
    print_result("maximum", h_maximum, count);
    print_result("minimum", h_minimum, count);
    print_result("modulus", h_modulus, count);

    CUDA_CHECK(cudaFree(d_input1));
    CUDA_CHECK(cudaFree(d_input2));
    CUDA_CHECK(cudaFree(d_multiply));
    CUDA_CHECK(cudaFree(d_divide));
    CUDA_CHECK(cudaFree(d_abs_diff));
    CUDA_CHECK(cudaFree(d_maximum));
    CUDA_CHECK(cudaFree(d_minimum));
    CUDA_CHECK(cudaFree(d_modulus));

    return 0;
}
#endif
