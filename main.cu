#include "lif.cuh"
#include "syn.cuh"
#include "netgen.hpp"
#include "network.hpp"
#include "model.hpp"
#include "time.h"
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

struct Net{
    NEUBlock* neurons;
    SYNBlock* syns;
};

__device__ void simulatePoisson(NEUBlock* block,int id,int step){
    block->Fired[id]=block->rate[id]>block->rand[id+step*block->num];
    if(block->Fired[id]){
        block->Fire_cnt[id]++;
        block->Last_fired[id]=step;
    }
}
__device__ inline real getInput(real* buffer,int id,int step){
    return buffer[id*MAX_DELAY+step%MAX_DELAY];
}
__device__ void simulateLIF(NEUBlock *block,int id,int step)
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
            block->I_exc[id] += getInput(block->I_buffer_exc,id,step);
            block->I_inh[id] += getInput(block->I_buffer_inh,id,step);
        }
    }
}
__constant__ real A_LTP=0.1;
__constant__ real A_LTD=-0.01;
__constant__ real TAU_LTP=17;
__constant__ real TAU_LTD=34;
__constant__ real W_max=40;
__constant__ real W_min=0;
__device__ inline void pushSpike(real* buffer,int nid,int step,real value){
    atomicAdd(&buffer[nid*MAX_DELAY+step],value);
}
__device__ void simulateSYN(SYNBlock* syns,NEUBlock* neurons,int ref,int step){
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



__global__ void simulateNeuron(Net* net,int step)
{
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    if(net->neurons->source[id]){
        simulatePoisson(net->neurons,id,step);
    }else{
        simulateLIF(net->neurons,id,step);
        
    }
    if(id==12){
        printf("[%d]%d'vm:%f\n",step,id,net->neurons->V_m[id]);
        // printf("[%d]%d'fired:%d\n",step,id,net->neurons->Fired[id]);
    }
    // printf("Hello World from GPU!\n");
    // printf("vm:%f\n",net->neurons->V_m[id]);
}
__global__ void simulateSynapse(Net* net,int step){
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    simulateSYN(net->syns,net->neurons,id,step);
    // printf("weight:%f\n",syns->weight[id]);
}
real random0001()
{
    return rand() % (999 + 1) / (float)(999 + 1);
}
Net* buildNetwork(Network& net,int steps){
    NEUBlock* neus=initLIFData(net.neurons.size(),steps);
    SYNBlock* syns=initSYNData(net.synapses.size());
    for(int i=0;i<net.neurons.size();i++){
        if(net.neurons[i]->source){
            neus->source[i]=true;
            neus->rate[i]=random0001();
            for(int j=0;j<steps;j++){
                neus->rand[i*steps+j]=random0001();
            }
        }else{
            neus->source[i]=false;
        }
    }
    syns->num=net.synapses.size();
    for(int i=0;i<net.synapses.size();i++){
        syns->src[i]=net.synapses[i]->src;
        syns->tar[i]=net.synapses[i]->tar;
        syns->delay[i]=net.synapses[i]->delay/Config::STEP;
        syns->weight[i]=net.synapses[i]->weight;
    }
    Net* cnet=new Net;
    cnet->neurons=neus;
    cnet->syns=syns;
    return cnet;
}
Net* copyNet2GPU(Net* cnet){
    NEUBlock* gneurons=copyLIF2GPU(cnet->neurons);
    SYNBlock* gsyns=copySYN2GPU(cnet->syns);
    Net* tmp=new Net;
    tmp->neurons=gneurons;
    tmp->syns=gsyns;
    Net* gnet=toGPU(tmp,1);
    delete tmp;
    return gnet;
}
void freeCNet(Net* cnet){
    freeLIFData(cnet->neurons);
    freeSYNData(cnet->syns);
    delete cnet;
}
void freeGNet(Net* gnet){
    Net* tmp=toCPU(gnet,1);
    freeGLIF(tmp->neurons);
    freeGSYN(tmp->syns);
    gpuFree(gnet);
}
void simulate(Network& net,int steps){
    Net* cnet=buildNetwork(net,steps);
    int ns=cnet->neurons->num;
    int ss=cnet->syns->num;
    Net* gnet=copyNet2GPU(cnet);
    for(int i=0;i<steps;i++){
        simulateNeuron<<<ns/1024+1,ns%1024>>>(gnet,i);
        simulateSynapse<<<ss/1024+1,ss%1024>>>(gnet,i);
    }
    std::cout<<"simulate complete"<<std::endl;
    freeCNet(cnet);
    freeGNet(gnet);
    cudaDeviceReset();
}

int main(void)
{
    srand((int)time(0));
    int steps=10;
    Model model;
    LIFConst* consts=new LIFConst;
    auto p1= model.createPop(10,true,consts);
    auto p2= model.createPop(10,false,consts);
    auto p3= model.createPop(10,false,consts);
    model.connect(p1,p2,{1,2},{0.002,0.008},0.5);
    model.connect(p2,p3,{1,2},{0.002,0.008},0.5);
    model.connect(p1,p3,{1,2},{0.002,0.008},0.5);
    Network net= NetGen::genNet(&model);
    simulate(net,steps);
    return 0;
}