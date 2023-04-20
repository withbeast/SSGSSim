#include <stdio.h>
#include "syn.cuh"

SYNBlock* initSYNData(int num){
    SYNBlock* block=new SYNBlock;
    block->num=num;
    block->src=new int[num]();
    block->tar=new int[num]();
    block->weight=new real[num]();
    block->delay=new int[num]();
    return block;
}
void freeSYNData(SYNBlock* block){
    delete [] block->src;
    delete [] block->tar;
    delete [] block->weight;
    delete [] block->delay;
    delete block;
}
SYNBlock* copySYN2GPU(SYNBlock* cblock){
    SYNBlock* gblock=nullptr;
    SYNBlock* tmp=new SYNBlock;
    int num=cblock->num;
    tmp->num=num;
    tmp->src=toGPU(cblock->src,num);
    tmp->tar=toGPU(cblock->tar,num);
    tmp->weight=toGPU(cblock->weight,num);
    tmp->delay=toGPU(cblock->delay,num);
    gblock=toGPU(tmp,1);
    delete tmp;
    return gblock;
}
void freeGSYN(SYNBlock* gblock){
    SYNBlock* tmp=toCPU(gblock,1);
    gpuFree(tmp->src);
    gpuFree(tmp->tar);
    gpuFree(tmp->weight);
    gpuFree(tmp->delay);
    gpuFree(gblock);
}
