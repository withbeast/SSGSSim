#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
typedef float real;
struct LIFBlock
{
    bool *Fired;
    int *FireCnt;
    real *V_m;
    real *I_exc;
    real *I_inh;
    int *Refrac_state;
    real* I_buffer_exc;
    real* I_buffer_inh;
};

struct LIFConst
{
    real P22;
    real P11exc;
    real P11inh;
    real P21exc;
    real P21inh;
    real V_reset;
    real V_rest;
    real C_m;
    real Tau_m;
    real Tau_exc;
    real Tau_inh;
    real V_thresh;
    real I_offset;
    real Refrac_step;
};
LIFBlock* initLIFData(int num);
void freeLIFData(LIFBlock* block);
LIFBlock* copy2GPU(LIFBlock* cblock,int num);
void freeGLIF(LIFBlock* gblock);
