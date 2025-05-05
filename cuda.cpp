#include <iostream>
#include <cuda_runtime.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vectorAdditionCUDA() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    size_t size = N * sizeof(float);
    A = new float[N];
    B = new float[N];
    C = new float[N];

    for (int i = 0; i < N; ++i) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Vector Addition Completed (Sample Output):\n";
    for (int i = 0; i < 10; ++i)
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#define N 512  // Size of square matrix

__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

void matrixMultiplicationCUDA() {
    int size = N * N * sizeof(float);
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = new float[N * N];
    B = new float[N * N];
    C = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "\nMatrix Multiplication Completed (Sample Output):\n";
    for (int i = 0; i < 10; ++i)
        std::cout << C[i * N + i] << " ";

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    vectorAdditionCUDA();
    matrixMultiplicationCUDA();
    return 0;
}
