#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>
typedef float real;

inline void CheckCall(cudaError_t err, const char *file, int line)
{
    const cudaError_t error = err;
    if (error != cudaSuccess)
    {
        printf("Error:%s.Line %d,", file, line);
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
#define CUDACHECK(x) CheckCall(x, __FILE__, __LINE__)

template<typename T>
T* toGPU(T* cpu, int size)
{
    T * ret;
    CUDACHECK(cudaMalloc((void**)&(ret), sizeof(T) * size));
    CUDACHECK(cudaMemcpy(ret, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));
    return ret;
}
template<typename T>
T* toCPU(T* gpu, int size)
{
	T * ret = static_cast<T*>(malloc(sizeof(T)*size));
	CUDACHECK(cudaMemcpy(ret, gpu, sizeof(T)*size, cudaMemcpyDeviceToHost));
	return ret;
}
template<typename T>
void gpuFree(T* gpu)
{
	CUDACHECK(cudaFree(gpu));
}