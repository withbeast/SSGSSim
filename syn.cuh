#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include "lif.cuh"
#include "mem.cuh"
typedef float real;
struct SYNBlock{
    int num;
    int* src;
    int* tar;
    real* weight;
    int* delay;
};
SYNBlock* initSYNData(int num);
void freeSYNData(SYNBlock* block);
SYNBlock* copySYN2GPU(SYNBlock* cblock);
void freeGSYN(SYNBlock* gblock);
