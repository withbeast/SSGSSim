
#include <stdio.h>
#include <string>
#include <vector>
#include "lif.cuh"

__constant__ real _P22 = 0.90483743;
__constant__ real _P11exc = 0.951229453;
__constant__ real _P11inh = 0.967216074;
__constant__ real _P21exc = 0.0799999982;
__constant__ real _P21inh = 0.0599999987;
__constant__ real V_rest = 0;
__constant__ real V_reset = 0;
__constant__ real C_m = 0.25;
__constant__ real Tau_m = 10.0;
__constant__ real V_thresh = 15;
__constant__ real I_offset = 0;
__constant__ real Refrac_step = 4;

__device__ void simulateLIF(LIFBlock *blocks,int id)
{
    blocks->Fired[id] = false;
    if (blocks->Refrac_state[id] > 0)
    {
        --blocks->Refrac_state[id];
    }
    else
    {
        blocks->V_m[id] = _P22 * blocks->V_m[id] + blocks->I_exc[id] * _P21exc + blocks->I_inh[id] * _P21inh;
        blocks->V_m[id] += (1 - _P22) * (I_offset * Tau_m / C_m + V_rest);
        blocks->I_exc[id] *= _P11exc;
        blocks->I_inh[id] *= _P11inh;
        if (blocks->V_m[id] >= V_thresh)
        {
            blocks->Fired[id] = true;
            blocks->FireCnt[id]++;
            blocks->V_m[id] = V_reset;
            blocks->Refrac_state[id] = Refrac_step;
        }
        else
        {
            blocks->I_exc[id] += blocks->I_syn_exc[id];
            blocks->I_inh[id] += blocks->I_syn_inh[id];
        }
    }
}

