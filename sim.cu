#include "lif.cuh"
#include "syn.cuh"
#include "net.cuh"
#include <stdio.h>
//LIF神经元使用常量
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

//STDP突触使用常量
__constant__ real A_LTP = 0.1;
__constant__ real A_LTD = -0.01;
__constant__ real TAU_LTP = 17;
__constant__ real TAU_LTD = 34;
__constant__ real W_max = 40;
__constant__ real W_min = 0;

//神经网络仿真使用常量
__constant__ int MAX_DELAY = 10;

__device__ void simulatePoisson(NEUBlock *block, int id, int step)
{
    block->Fired[id] = block->rate[id] > block->rand[id + step * block->num];
    if (block->Fired[id])
    {
        block->Fire_cnt[id]++;
        block->Last_fired[id] = step;
    }
}
__device__ inline real getInput(real *buffer, int id, int step)
{
    return buffer[id * MAX_DELAY + step % MAX_DELAY];
}
__device__ inline void pushSpike(real *buffer, int nid, int step, real value)
{
    atomicAdd(&buffer[nid * MAX_DELAY + step], value);
}
__device__ void simulateLIF(NEUBlock *block, int id, int step)
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
            block->Fire_cnt[id]++;
            block->V_m[id] = V_reset;
            block->Refrac_state[id] = Refrac_step;
        }
        else
        {
            block->I_exc[id] += getInput(block->I_buffer_exc, id, step);
            block->I_inh[id] += getInput(block->I_buffer_inh, id, step);
        }
    }
}
__device__ void simulateSYN(SYNBlock *syns, NEUBlock *neurons, int ref, int step)
{
    int src = syns->src[ref];
    int tar = syns->tar[ref];
    real weight = syns->weight[ref];
    int delay = syns->delay[ref];
    if (!neurons->Fired[src] && !neurons->Fired[tar])
        return;
    // 发放脉冲
    if (neurons->Fired[syns->src[ref]])
    {
        if (weight > 0)
            pushSpike(neurons->I_buffer_exc, tar, (step + delay) % MAX_DELAY, weight);
        else
            pushSpike(neurons->I_buffer_inh, tar, (step + delay) % MAX_DELAY, weight);
    }
    // STDP权重更新
    int dt = neurons->Last_fired[src] - neurons->Last_fired[tar];
    real dw = 0;
    if (dt < 0)
    {
        dw = A_LTP * exp(dt / TAU_LTP);
    }
    else if (dt > 0)
    {
        dw = A_LTD * exp(-dt / TAU_LTD);
    }
    else
    {
        dw = 0;
    }
    real nweight = syns->weight[ref] + dw;
    nweight = (nweight > W_max) ? W_max : ((nweight < W_min) ? W_min : nweight);
    atomicExch(&(syns->weight[ref]), nweight);
}


__global__ void simulateNeuron(Net *net, int step)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net->neurons->source[id])
    {
        simulatePoisson(net->neurons, id, step);
    }
    else
    {
        simulateLIF(net->neurons, id, step);
    }
    if (id == 12)
    {
        // printf("[%d]%d'vm:%f\n", step, id, net->neurons->V_m[id]);
        // printf("[%d]%d'fired:%d\n",step,id,net->neurons->Fired[id]);
    }
    // printf("Hello World from GPU!\n");
    // printf("vm:%f\n",net->neurons->V_m[id]);
}
__global__ void simulateSynapse(Net *net, int step)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    simulateSYN(net->syns, net->neurons, id, step);
    // if(id==15)
    // printf("weight:%f\n",net->syns->weight[id]);
}


void simulate1(Net* cnet, int steps)
{
    int ns = cnet->neurons->num;
    int ss = cnet->syns->num;
    Net *gnet = copyNet2GPU(cnet);
    for (int i = 0; i < steps; i++)
    {
        simulateNeuron<<<ns / 1024 + 1, ns % 1024>>>(gnet, i);
        simulateSynapse<<<ss / 1024 + 1, ss % 1024>>>(gnet, i);

    }
    printf("simulate1 complete\n");
    freeCNet(cnet);
    freeGNet(gnet);
    cudaDeviceReset();
}



__device__ void simulatePoissonVAR(NEUBlockVAR *block, int id, int step)
{
    block->Fired[id] = block->rate[id] > block->rand[id + step * block->num];
    if (block->Fired[id])
    {
        block->Fire_cnt[id]++;
        block->Last_fired[id] = step;
    }
}
__device__ void simulateLIFVAR(NEUBlockVAR *block, int id, int step)
{
    block->Fired[id] = false;
    if (block->Refrac_state[id] > 0)
    {
        --block->Refrac_state[id];
    }
    else
    {
        block->V_m[id] = block->_P22[id] * block->V_m[id] + block->I_exc[id] * block->_P21exc[id] + block->I_inh[id] * block->_P21inh[id];
        block->V_m[id] += (1 - block->_P22[id]) * (block->I_offset[id] * block->Tau_m[id] / block->C_m[id] + block->V_rest[id]);
        block->I_exc[id] *= block->_P11exc[id];
        block->I_inh[id] *= block->_P11inh[id];
        if (block->V_m[id] >= block->V_thresh[id])
        {
            block->Fired[id] = true;
            block->Fire_cnt[id]++;
            block->V_m[id] = block->V_reset[id];
            block->Refrac_state[id] = block->Refrac_step[id];
        }
        else
        {
            block->I_exc[id] += getInput(block->I_buffer_exc, id, step);
            block->I_inh[id] += getInput(block->I_buffer_inh, id, step);
        }
    }
}

__device__ void simulateSYNVAR(SYNBlockVAR *syns, NEUBlockVAR *neurons, int ref, int step)
{
    int src = syns->src[ref];
    int tar = syns->tar[ref];
    real weight = syns->weight[ref];
    int delay = syns->delay[ref];
    if (!neurons->Fired[src] && !neurons->Fired[tar])
        return;
    // 发放脉冲
    if (neurons->Fired[syns->src[ref]])
    {
        if (weight > 0)
            pushSpike(neurons->I_buffer_exc, tar, (step + delay) % MAX_DELAY, weight);
        else
            pushSpike(neurons->I_buffer_inh, tar, (step + delay) % MAX_DELAY, weight);
    }
    // STDP权重更新
    int dt = neurons->Last_fired[src] - neurons->Last_fired[tar];
    real dw = 0;
    if (dt < 0)
    {
        dw = syns->A_LTP[ref] * exp(dt / syns->TAU_LTP[ref]);
    }
    else if (dt > 0)
    {
        dw = syns->A_LTD[ref] * exp(-dt / syns->TAU_LTD[ref]);
    }
    else
    {
        dw = 0;
    }
    real nweight = syns->weight[ref] + dw;
    nweight = (nweight > syns->W_max[ref]) ? syns->W_max[ref] : ((nweight < syns->W_min[ref]) ? syns->W_min[ref] : nweight);
    atomicExch(&(syns->weight[ref]), nweight);
}
__global__ void simulateNeuronVAR(NetVAR *net, int step)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net->neurons->source[id])
    {
        simulatePoissonVAR(net->neurons, id, step);
    }
    else
    {
        simulateLIFVAR(net->neurons, id, step);
    }
    if (id == 12)
    {
        // printf("[%d]%d'vm:%f\n", step, id, net->neurons->V_m[id]);
        // printf("[%d]%d'fired:%d\n",step,id,net->neurons->Fired[id]);
    }
    // printf("Hello World from GPU!\n");
    // printf("vm:%f\n",net->neurons->V_m[id]);
}
__global__ void simulateSynapseVAR(NetVAR *net, int step)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    simulateSYNVAR(net->syns, net->neurons, id, step);
    // if(id==15)
    // printf("weight:%f\n",net->syns->weight[id]);
}


void simulate2(NetVAR *cnet, int steps)
{
    int ns = cnet->neurons->num;
    int ss = cnet->syns->num;
    NetVAR *gnet = copyNet2GPU(cnet);
    for (int i = 0; i < steps; i++)
    {
        simulateNeuronVAR<<<ns / 1024 + 1, ns % 1024>>>(gnet, i);
        simulateSynapseVAR<<<ss / 1024 + 1, ss % 1024>>>(gnet, i);

    }
    printf("simulate2 complete\n");
    freeCNet(cnet);
    freeGNet(gnet);
    cudaDeviceReset();
}