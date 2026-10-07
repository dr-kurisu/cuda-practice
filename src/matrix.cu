#include <cuda_runtime.h>
#include <memory.h>
#include <cstdlib>
#include <cuda/cmath>


void initMatrix(float* A, int N) {
    for (int i = 0; i < N; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialMatrixMultiple(float* A, float* B, float* C, int M, int N, int O)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < O; j++)
        {
            C[i * O + j] = 0.0f;
            for (int k = 0; k < N; k++)
            {
                C[i * O + j] += A[i * N + k] * B[k * O + j];
            }
        }
    }
}

__global__
void naiveCudaMatrixMultiple(float* A, float* B, float* C, int M, int N, int O) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || j >= O)
            return;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * O + j];
    }
    C[i * O + j] = sum;
}

bool ApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    int M = 1000;
    int N = 700;
    int O = 1200;

    float* A = (float*)malloc(sizeof(float) * M * N);
    float* B = (float*)malloc(sizeof(float) * N * O);
    float* C = (float*)malloc(sizeof(float) * M * O);

    initMatrix(A, M * N);
    initMatrix(B, N * O);

    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    cudaMalloc(&devA, sizeof(float) * M * N);
    cudaMalloc(&devB, sizeof(float) * N * O);
    cudaMalloc(&devC, sizeof(float) * M * O);

    cudaMemcpy(devA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, sizeof(float) * N * O, cudaMemcpyHostToDevice);
    cudaMemset(devC, 0, sizeof(float) * M * O);

    serialMatrixMultiple(A, B, C, M, N, O);

    dim3 block(16, 16);
    dim3 grid(
        cuda::ceil_div(O, block.x),
        cuda::ceil_div(M, block.y)
    );

    // lanuch kernel
    naiveCudaMatrixMultiple<<<grid, block>>>(devA, devB, devC, M, N, O);
    cudaDeviceSynchronize();

    float* result = (float*)malloc(sizeof(float) * M * O);
    cudaMemcpy(result, devC, sizeof(float) * M * O, cudaMemcpyDeviceToHost);


    if(ApproximatelyEqual(C, result, M * O, 1e-4f))
    {
        printf("CPU and GPU answers match\n");
    }
    else
    {
        printf("CPU and GPU answers to not match\n");
    }

    return 0;
}
