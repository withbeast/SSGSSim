#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
typedef float real;
struct NEUBlock
{
    int num;
    ///LIF
    bool *Fired;
    int *Fire_cnt;
    int *Last_fired;
    real *V_m;
    real *I_exc;
    real *I_inh;
    int *Refrac_state;
    real *I_buffer_exc;
    real *I_buffer_inh;
    ///泊松
    int steps;
    bool* source;
    real *rand;
    real *rate;
};

struct GLIFConst
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
NEUBlock *initLIFData(int num,int steps);
void freeLIFData(NEUBlock *block);
NEUBlock *copyLIF2GPU(NEUBlock *cblock);
void freeGLIF(NEUBlock *gblock);
