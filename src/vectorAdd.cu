#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>
#include <stdio.h>
// #include <cuda/cmath>


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

void unifiedMemExample(int length) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparsion = (float*)malloc(length * sizeof(float));

    // unified memory can be both accessed by host and device
    cudaMallocManaged(&A, length * sizeof(float));
    cudaMallocManaged(&B, length * sizeof(float));
    cudaMallocManaged(&C, length * sizeof(float));

    initArray(A, length);
    initArray(B, length);

    int thread = 256;
    int block = std::ceil(length / thread);
    vecAdd<<<thread, block>>>(A, B, C, length);

    // sync
    cudaDeviceSynchronize();

    serialAdd(A, B, comparsion, length);

    if (vectorApproximatelyEqual(C, comparsion, length)) {
        std::cout << "Correct" << std::endl;
    }else {
        std::cout << "Wrong" << std::endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparsion);
}

int main(int argc, char** argv) {
    int length = 1024;
    if (argc >= 2) {
        length = std::atoi(argv[1]);
    }
    unifiedMemExample(length);

    return 0;
}
