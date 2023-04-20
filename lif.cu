
#include <stdio.h>
#include "lif.cuh"
#include "mem.cuh"
const int Max_delay=10;
NEUBlock* initLIFData(int num,int steps){
    NEUBlock* block=new NEUBlock;
    block->num=num;
    block->V_m=new real[num]();
    block->Fired=new bool[num]();
    block->Fire_cnt=new int[num]();
    block->Last_fired=new int[num]();
    block->I_exc=new real[num]();
    block->I_inh=new real[num]();
    block->I_buffer_exc=new real[num*Max_delay]();
    block->I_buffer_inh=new real[num*Max_delay]();
    block->Refrac_state=new int[num]();

    block->steps=steps;
    block->source=new bool[num]();
    block->rand=new real[num*steps]();
    block->rate=new real[num]();
    return block;
}
void freeLIFData(NEUBlock* block){
    delete[] block->V_m;
    delete[] block->Fire_cnt;
    delete[] block->Fired;
    delete[] block->Last_fired;
    delete[] block->I_exc;
    delete[] block->I_inh;
    delete[] block->I_buffer_exc;
    delete[] block->I_buffer_inh;
    delete[] block->Refrac_state;

    delete[] block->rand;
    delete[] block->rate;
    delete[] block->source;
    delete block;
}
NEUBlock* copyLIF2GPU(NEUBlock* cblock){
    NEUBlock* gblock;
    NEUBlock* tmp=new NEUBlock;
    int num=cblock->num;
    tmp->num=num;
    
    tmp->V_m=toGPU(cblock->V_m,num);
    tmp->I_exc=toGPU(cblock->I_exc,num);
    tmp->I_inh=toGPU(cblock->I_inh,num);
    tmp->I_buffer_exc=toGPU(cblock->I_buffer_exc,num*Max_delay);
    tmp->I_buffer_inh=toGPU(cblock->I_buffer_inh,num*Max_delay);
    tmp->Fired=toGPU(cblock->Fired,num);
    tmp->Fire_cnt=toGPU(cblock->Fire_cnt,num);
    tmp->Last_fired=toGPU(cblock->Last_fired,num);
    tmp->Refrac_state=toGPU(cblock->Refrac_state,num);

    int steps=cblock->steps;
    tmp->steps=steps;
    tmp->rate=toGPU(cblock->rate,num);
    tmp->source=toGPU(cblock->source,num);
    tmp->rand=toGPU(cblock->rand,num*steps);

    gblock=toGPU(tmp,1);
    return gblock;
}
void freeGLIF(NEUBlock* gblock){
    NEUBlock* tmp=toCPU(gblock,1);
    gpuFree(tmp->V_m);
    gpuFree(tmp->I_exc);
    gpuFree(tmp->I_inh);
    gpuFree(tmp->I_buffer_exc);
    gpuFree(tmp->I_buffer_inh);
    gpuFree(tmp->Fired);
    gpuFree(tmp->Fire_cnt);
    gpuFree(tmp->Last_fired);
    gpuFree(tmp->Refrac_state);

    gpuFree(tmp->rand);
    gpuFree(tmp->rate);
    gpuFree(tmp->source);
    gpuFree(gblock);
    delete tmp;
}





