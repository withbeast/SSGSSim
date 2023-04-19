
#include <stdio.h>
#include "lif.cuh"
#include "mem.cuh"
const int Max_delay=10;
LIFBlock* initLIFData(int num){
    LIFBlock* block=new LIFBlock;
    block->V_m=new real[num]();
    block->Fired=new bool[num]();
    block->FireCnt=new int[num]();
    block->I_exc=new real[num]();
    block->I_inh=new real[num]();
    block->I_buffer_exc=new real[num*Max_delay]();
    block->I_buffer_inh=new real[num*Max_delay]();
    block->Refrac_state=new int[num]();
    return block;
}
void freeLIFData(LIFBlock* block){
    delete[] block->V_m;
    delete[] block->FireCnt;
    delete[] block->Fired;
    delete[] block->I_exc;
    delete[] block->I_inh;
    delete[] block->I_buffer_exc;
    delete[] block->I_buffer_inh;
    delete[] block->Refrac_state;
    delete block;
}
LIFBlock* copy2GPU(LIFBlock* cblock,int num){
    LIFBlock* gblock;
    LIFBlock* tmp=new LIFBlock;
    tmp->V_m=toGPU(cblock->V_m,num);
    tmp->I_exc=toGPU(cblock->I_exc,num);
    tmp->I_inh=toGPU(cblock->I_inh,num);
    tmp->I_buffer_exc=toGPU(cblock->I_buffer_exc,num*Max_delay);
    tmp->I_buffer_inh=toGPU(cblock->I_buffer_inh,num*Max_delay);
    tmp->Fired=toGPU(cblock->Fired,num);
    tmp->FireCnt=toGPU(cblock->FireCnt,num);
    tmp->Refrac_state=toGPU(cblock->Refrac_state,num);
    gblock=toGPU(tmp,1);
    return gblock;
}
void freeGLIF(LIFBlock* gblock){
    // LIFBlock* tmp=new LIFBlock;
    // cudaMemcpy(tmp,gblock,sizeof(LIFBlock),cudaMemcpyDeviceToHost);
    LIFBlock* tmp=toCPU(gblock,1);
    gpuFree(tmp->V_m);
    gpuFree(tmp->I_exc);
    gpuFree(tmp->I_inh);
    gpuFree(tmp->I_buffer_exc);
    gpuFree(tmp->I_buffer_inh);
    gpuFree(tmp->Fired);
    gpuFree(tmp->FireCnt);
    gpuFree(tmp->Refrac_state);
    gpuFree(gblock);
    delete tmp;
}





