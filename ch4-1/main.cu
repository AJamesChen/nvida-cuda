/*
Create a program based on the prime number testing program but updating two arrays: the first with the number tested by the thread and the second, on the same index position, to hold the test result (whether or not the number is prime). Copy the results back to the host and print only the prime numbers.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__host__ __device__ int is_prime_number(int value) {
    if (value <= 1) {
        return 0;
    }

    if (value == 2) {
        return 1;
    }

    if (value % 2 == 0) {
        return 0;
    }

    int j = 3;
    while (j <= value / j) {
        if (value % j == 0) {
            return 0;
        }
        j += 2;
    }

    return 1;
}

__global__ void is_prime(int *d_input, int *d_output, int count) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < count) {
        d_output[i] = is_prime_number(d_input[i]);
    }
}


#ifndef UNIT_TEST
int main() {
    int count = 10;
    int h_input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int h_output[10];

    int *d_input, *d_output;
    cudaMalloc(&d_input, count * sizeof(int));
    cudaMalloc(&d_output, count * sizeof(int));

    cudaMemcpy(d_input, h_input, count * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;
    is_prime<<<numBlocks, blockSize>>>(d_input, d_output, count);

    cudaMemcpy(h_output, d_output, count * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; i++) {
        printf("%2d is %s\n", h_input[i], h_output[i] ? "prime" : "not prime");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
#endif
