#include "lif.cuh"
#include "syn.cuh"
#include <iostream>
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
__constant__ int MAX_DELAY=10;
__device__ inline real getInput(real* buffer,int id,int step){
    return buffer[id*MAX_DELAY+step%MAX_DELAY];
}
__device__ void simulateLIF(LIFBlock *block,int id,int step)
{
    block->Fired[id] = false;
    if (block->Refrac_state[id] > 0)
    {
        --block->Refrac_state[id];
    }
    else
    {
        block->V_m[id] = _P22 * block->V_m[id] + block->I_exc[id] * _P21exc + block->I_inh[id] * _P21inh;
        block->V_m[id] += (1 - _P22) * (I_offset * Tau_m / C_m + V_rest);
        block->I_exc[id] *= _P11exc;
        block->I_inh[id] *= _P11inh;
        if (block->V_m[id] >= V_thresh)
        {
            block->Fired[id] = true;
            block->FireCnt[id]++;
            block->V_m[id] = V_reset;
            block->Refrac_state[id] = Refrac_step;
        }
        else
        {
            block->I_exc[id] += getInput(block->I_buffer_exc,id,step);
            block->I_inh[id] += getInput(block->I_buffer_inh,id,step);
        }
    }
}
__device__ inline void pushSpike(real* buffer,int nid,int step,real value){
    atomicAdd(&buffer[nid*MAX_DELAY+step],value);
}
__device__ void simulateSYN(SYNBlock* syns,LIFBlock* neurons,int ref,int step){
    int src=syns->src[ref];
    int tar=syns->tar[ref];
    real weight=syns->weight[ref];  
    int delay=syns->delay[ref];
    if(!neurons->Fired[src]&&!neurons->Fired[tar])return;
    //STDP权重更新
    //发放脉冲
    if(neurons->Fired[syns->src[ref]]){
        if(weight>0)
            pushSpike(neurons->I_buffer_exc,tar,(step+delay)%MAX_DELAY,weight);
        else
            pushSpike(neurons->I_buffer_inh,tar,(step+delay)%MAX_DELAY,weight);
    }
}


__global__ void simulateNeuron(LIFBlock* neurons,int step)
{
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    simulateLIF(neurons,id,step);
    
    // printf("Hello World from GPU!\n");
    printf("vm:%f\n",neurons->V_m[id]);
}
__global__ void simulateSynapse(LIFBlock* neurons,SYNBlock* syns,int step){
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    simulateSYN(syns,neurons,id,step);
    // printf("weight:%f\n",syns->weight[id]);
}

int main(void)
{
    LIFBlock* neurons=initLIFData(10);
    SYNBlock* syns=initSYNData(20);


    LIFBlock* gneurons=copy2GPU(neurons,10);
    SYNBlock* gsyns=copy2GPU(syns,10);
    int steps=1;
    for(int i=0;i<steps;i++){
        simulateNeuron<<<1,10>>>(gneurons,0);
        simulateSynapse<<<1,20>>>(gneurons,gsyns,0);
    }
    std::cout<<"simulate complete"<<std::endl;
    freeLIFData(neurons);
    freeSYNData(syns);
    freeGLIF(gneurons);
    freeGSYN(gsyns);
    cudaDeviceReset();
    return 0;
}