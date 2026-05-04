/*
Create a program that performs the scalar multiplication of a vector. It receives as input parameters a float number and a float array, and executes the multiplication of each element of the array by the given number. Remember to copy the result back to the host
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void multiply_array(float d_input_1, float *d_input_2, float *d_output, int count) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = d_input_1 * d_input_2[i];
    }
}


#ifndef UNIT_TEST
int main() {
    int count = 10;
    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, sizeof(float));
    cudaMalloc(&d_input2, count * sizeof(float));
    cudaMalloc(&d_output, count * sizeof(float));

    float h_input1 = 2.5f;
    float h_input2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float h_output[10];

    cudaMemcpy(d_input1, &h_input1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, count * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;
    multiply_array<<<numBlocks, blockSize>>>(h_input1, d_input2, d_output, count);

    cudaMemcpy(h_output, d_output, count * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; i++) {
        printf("%2f x %2f is %2f\n", h_input1, h_input2[i], h_output[i]);
    }

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);

    return 0;
}
#endif
