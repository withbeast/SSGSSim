#include "lif.cuh"
const int Max_delay = 10;
NEUBlock *initLIFData(int num, int steps)
{
    NEUBlock *block = new NEUBlock;
    block->num = num;
    block->V_m = new real[num]();
    block->Fired = new bool[num]();
    block->Fire_cnt = new int[num]();
    block->Last_fired = new int[num]();
    block->I_exc = new real[num]();
    block->I_inh = new real[num]();
    block->I_buffer_exc = new real[num * Max_delay]();
    block->I_buffer_inh = new real[num * Max_delay]();
    block->Refrac_state = new int[num]();

    block->steps = steps;
    block->source = new bool[num]();
    block->rand = new real[num * steps]();
    block->rate = new real[num]();
    return block;
}


void freeLIFData(NEUBlock *block)
{
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
NEUBlock *copyLIF2GPU(NEUBlock *cblock)
{
    NEUBlock *gblock;
    NEUBlock *tmp = new NEUBlock;
    int num = cblock->num;
    tmp->num = num;

    tmp->V_m = toGPU(cblock->V_m, num);
    tmp->I_exc = toGPU(cblock->I_exc, num);
    tmp->I_inh = toGPU(cblock->I_inh, num);
    tmp->I_buffer_exc = toGPU(cblock->I_buffer_exc, num * Max_delay);
    tmp->I_buffer_inh = toGPU(cblock->I_buffer_inh, num * Max_delay);
    tmp->Fired = toGPU(cblock->Fired, num);
    tmp->Fire_cnt = toGPU(cblock->Fire_cnt, num);
    tmp->Last_fired = toGPU(cblock->Last_fired, num);
    tmp->Refrac_state = toGPU(cblock->Refrac_state, num);

    int steps = cblock->steps;
    tmp->steps = steps;
    tmp->rate = toGPU(cblock->rate, num);
    tmp->source = toGPU(cblock->source, num);
    tmp->rand = toGPU(cblock->rand, num * steps);

    gblock = toGPU(tmp, 1);
    return gblock;
}
void freeGLIF(NEUBlock *gblock)
{
    NEUBlock *tmp = toCPU(gblock, 1);
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



// __constant__ real _P22 = 0.90483743;
// __constant__ real _P11exc = 0.951229453;
// __constant__ real _P11inh = 0.967216074;
// __constant__ real _P21exc = 0.0799999982;
// __constant__ real _P21inh = 0.0599999987;
// __constant__ real V_rest = 0;
// __constant__ real V_reset = 0;
// __constant__ real C_m = 0.25;
// __constant__ real Tau_m = 10.0;
// __constant__ real V_thresh = 15;
// __constant__ real I_offset = 0;
// __constant__ real Refrac_step = 4;
NEUBlockVAR *initLIFDataVAR(int num, int steps)
{
    NEUBlockVAR *block = new NEUBlockVAR;
    block->num = num;
    block->V_m = new real[num]();
    block->Fired = new bool[num]();
    block->Fire_cnt = new int[num]();
    block->Last_fired = new int[num]();
    block->I_exc = new real[num]();
    block->I_inh = new real[num]();
    block->I_buffer_exc = new real[num * Max_delay]();
    block->I_buffer_inh = new real[num * Max_delay]();
    block->Refrac_state = new int[num]();

    block->steps = steps;
    block->source = new bool[num]();
    block->rand = new real[num * steps]();
    block->rate = new real[num]();

    block->_P22 = new real[num]();
    std::fill(block->_P22, block->_P22 + num, 0.90483743);
    block->_P11exc = new real[num]();
    std::fill(block->_P11exc, block->_P11exc + num, 0.951229453);
    block->_P11inh = new real[num]();
    std::fill(block->_P11inh, block->_P11inh + num, 0.967216074);
    block->_P21exc = new real[num]();
    std::fill(block->_P21exc, block->_P21exc + num, 0.0799999982);
    block->_P21inh = new real[num]();
    std::fill(block->_P21inh, block->_P21inh + num, 0.0599999987);
    block->V_rest = new real[num]();
    std::fill(block->V_rest, block->V_rest + num, 0);
    block->V_reset = new real[num]();
    std::fill(block->V_reset, block->V_reset + num, 0);
    block->C_m = new real[num]();
    std::fill(block->C_m, block->C_m + num, 0.25);
    block->Tau_m = new real[num]();
    std::fill(block->Tau_m, block->Tau_m + num, 10.0);
    block->V_thresh = new real[num]();
    std::fill(block->V_thresh, block->V_thresh + num, 15);
    block->I_offset = new real[num]();
    std::fill(block->I_offset, block->I_offset + num, 0);
    block->Refrac_step = new real[num]();
    std::fill(block->Refrac_step, block->Refrac_step + num, 4);
    return block;
}
void freeLIFData(NEUBlockVAR *block)
{
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

    delete[] block->_P22;
    delete[] block->_P21exc;
    delete[] block->_P21inh;
    delete[] block->_P11exc;
    delete[] block->_P11inh;
    delete[] block->V_rest;
    delete[] block->V_reset;
    delete[] block->C_m;
    delete[] block->Tau_m;
    delete[] block->V_thresh;
    delete[] block->I_offset;
    delete[] block->Refrac_step;
    delete block;
}
NEUBlockVAR *copyLIF2GPU(NEUBlockVAR *cblock)
{
    NEUBlockVAR *gblock;
    NEUBlockVAR *tmp = new NEUBlockVAR;
    int num = cblock->num;
    tmp->num = num;

    tmp->V_m = toGPU(cblock->V_m, num);
    tmp->I_exc = toGPU(cblock->I_exc, num);
    tmp->I_inh = toGPU(cblock->I_inh, num);
    tmp->I_buffer_exc = toGPU(cblock->I_buffer_exc, num * Max_delay);
    tmp->I_buffer_inh = toGPU(cblock->I_buffer_inh, num * Max_delay);
    tmp->Fired = toGPU(cblock->Fired, num);
    tmp->Fire_cnt = toGPU(cblock->Fire_cnt, num);
    tmp->Last_fired = toGPU(cblock->Last_fired, num);
    tmp->Refrac_state = toGPU(cblock->Refrac_state, num);

    int steps = cblock->steps;
    tmp->steps = steps;
    tmp->rate = toGPU(cblock->rate, num);
    tmp->source = toGPU(cblock->source, num);
    tmp->rand = toGPU(cblock->rand, num * steps);


    tmp->_P22=toGPU(cblock->_P22,num);
    tmp->_P21exc=toGPU(cblock->_P21exc,num);
    tmp->_P21inh=toGPU(cblock->_P21inh,num);
    tmp->_P11exc=toGPU(cblock->_P11exc,num);
    tmp->_P11inh=toGPU(cblock->_P11inh,num);
    tmp->V_rest=toGPU(cblock->V_rest,num);
    tmp->V_reset=toGPU(cblock->V_reset,num);
    tmp->C_m=toGPU(cblock->C_m,num);
    tmp->Tau_m=toGPU(cblock->Tau_m,num);
    tmp->V_thresh=toGPU(cblock->V_thresh,num);
    tmp->I_offset=toGPU(cblock->I_offset,num);
    tmp->Refrac_step=toGPU(cblock->Refrac_step,num);

    gblock = toGPU(tmp, 1);
    return gblock;
}
void freeGLIF(NEUBlockVAR *gblock)
{
    NEUBlockVAR *tmp = toCPU(gblock, 1);
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

    gpuFree(tmp->_P22);
    gpuFree(tmp->_P21exc);
    gpuFree(tmp->_P21inh);
    gpuFree(tmp->_P11exc);
    gpuFree(tmp->_P11inh);
    gpuFree(tmp->V_reset);
    gpuFree(tmp->V_rest);
    gpuFree(tmp->C_m);
    gpuFree(tmp->Tau_m);
    gpuFree(tmp->V_thresh);
    gpuFree(tmp->I_offset);
    gpuFree(tmp->Refrac_step);


    gpuFree(gblock);
    delete tmp;
}