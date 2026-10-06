#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>


__global__ void vecAdd(float* A, float* B, float* C, int length) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < length) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length) {
    std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialAdd(float* A, float* B, float* C, int length) {
    for (int i = 0; i < length; i++) {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
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

void explicitMemoryExample(int length) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    float* comparsion = (float*)malloc(sizeof(float) * length);

    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    cudaMallocHost(&A, length*sizeof(float));
    cudaMallocHost(&B, length*sizeof(float));
    cudaMallocHost(&C, length*sizeof(float));

    initArray(A, length);
    initArray(B, length);

    cudaMalloc(&devA, length*sizeof(float));
    cudaMalloc(&devB, length*sizeof(float));
    cudaMalloc(&devC, length*sizeof(float));

    // copy from host to device
    cudaMemcpy(devA, A, length*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, length*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, length*sizeof(float));

    int threads = 256;
    int blocks = cuda::ceil_div(length, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, length);

    cudaDeviceSynchronize();
    cudaMemcpy(C, devC, length*sizeof(float), cudaMemcpyDefault);

    serialAdd(A, B, comparsion, length);

    if(vectorApproximatelyEqual(C, comparsion, length))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparsion);
}

int main(int argc, char** argv) {
    int length = 1024;
    if (argc >= 2) {
        length = std::atoi(argv[1]);
    }
    explicitMemoryExample(length);

    return 0;
}
