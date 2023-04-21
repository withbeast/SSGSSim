#pragma once

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


struct SYNBlockVAR{
    int num;
    int* src;
    int* tar;
    real* weight;
    int* delay;
    real* A_LTP;
    real* A_LTD;
    real* TAU_LTP;
    real* TAU_LTD;
    real* W_max;
    real* W_min;
};
SYNBlockVAR* initSYNDataVAR(int num);
void freeSYNData(SYNBlockVAR* block);
SYNBlockVAR* copySYN2GPU(SYNBlockVAR* cblock);
void freeGSYN(SYNBlockVAR* gblock);
