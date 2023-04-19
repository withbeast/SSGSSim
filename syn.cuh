#include <stdio.h>
#include <cuda_runtime.h>
#include "lif.cuh"
#include "mem.cuh"
typedef float real;
struct SYNBlock{
    int* src;
    int* tar;
    real* weight;
    int* delay;
};
SYNBlock* initSYNData(int num);
void freeSYNData(SYNBlock* block);
SYNBlock* copy2GPU(SYNBlock* cblock,int num);
void freeGSYN(SYNBlock* gblock);
