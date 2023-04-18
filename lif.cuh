#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
typedef float real;
struct LIFBlock
{
    int n;
    bool *Fired;
    int *FireCnt;
    real *V_m;
    real *I_syn_exc;
    real *I_syn_inh;
    real *I_exc;
    real *I_inh;
    int *Refrac_state;
};
struct SYNBlock{

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
__device__ void simulateLIF(LIFBlock *blocks,int id);