#include "syn.cuh"

SYNBlock *initSYNData(int num)
{
    SYNBlock *block = new SYNBlock;
    block->num = num;
    block->src = new int[num]();
    block->tar = new int[num]();
    block->weight = new real[num]();
    block->delay = new int[num]();
    return block;
}

void freeSYNData(SYNBlock *block)
{
    delete[] block->src;
    delete[] block->tar;
    delete[] block->weight;
    delete[] block->delay;
    delete block;
}

SYNBlock *copySYN2GPU(SYNBlock *cblock)
{
    SYNBlock *gblock = nullptr;
    SYNBlock *tmp = new SYNBlock;
    int num = cblock->num;
    tmp->num = num;
    tmp->src = toGPU(cblock->src, num);
    tmp->tar = toGPU(cblock->tar, num);
    tmp->weight = toGPU(cblock->weight, num);
    tmp->delay = toGPU(cblock->delay, num);
    gblock = toGPU(tmp, 1);
    delete tmp;
    return gblock;
}

void freeGSYN(SYNBlock *gblock)
{
    SYNBlock *tmp = toCPU(gblock, 1);
    gpuFree(tmp->src);
    gpuFree(tmp->tar);
    gpuFree(tmp->weight);
    gpuFree(tmp->delay);
    gpuFree(gblock);
}
/*
* 使用变量的形式存储不变量
*/

SYNBlockVAR *initSYNDataVAR(int num)
{
    SYNBlockVAR *block = new SYNBlockVAR;
    block->num = num;
    block->src = new int[num]();
    block->tar = new int[num]();
    block->weight = new real[num]();
    block->delay = new int[num]();

    block->A_LTP = new real[num]();
    std::fill(block->A_LTP, block->A_LTP + num, 0.1);
    block->A_LTD = new real[num]();
    std::fill(block->A_LTD, block->A_LTD + num, -0.01);
    block->TAU_LTP = new real[num]();
    std::fill(block->TAU_LTP, block->TAU_LTP + num, 17);
    block->TAU_LTD = new real[num]();
    std::fill(block->TAU_LTD, block->TAU_LTD + num, 34);
    block->W_max = new real[num]();
    std::fill(block->W_max, block->W_max + num, 40);
    block->W_min = new real[num]();
    std::fill(block->W_min, block->W_min + num, 0);
    return block;
}
SYNBlockVAR *copySYN2GPU(SYNBlockVAR *cblock)
{
    SYNBlockVAR *gblock = nullptr;
    SYNBlockVAR *tmp = new SYNBlockVAR;
    int num = cblock->num;
    tmp->num = num;
    tmp->src = toGPU(cblock->src, num);
    tmp->tar = toGPU(cblock->tar, num);
    tmp->weight = toGPU(cblock->weight, num);
    tmp->delay = toGPU(cblock->delay, num);
    tmp->A_LTP=toGPU(cblock->A_LTP,num);
    tmp->A_LTD=toGPU(cblock->A_LTD,num);
    tmp->TAU_LTD=toGPU(cblock->TAU_LTD,num);
    tmp->TAU_LTP=toGPU(cblock->TAU_LTP,num);
    tmp->W_max=toGPU(cblock->W_max,num);
    tmp->W_min=toGPU(cblock->W_min,num);
    gblock = toGPU(tmp, 1);
    delete tmp;
    return gblock;
}
void freeSYNData(SYNBlockVAR *block)
{
    delete[] block->src;
    delete[] block->tar;
    delete[] block->weight;
    delete[] block->delay;
    delete[] block->A_LTP;
    delete[] block->A_LTD;
    delete[] block->TAU_LTD;
    delete[] block->TAU_LTP;
    delete[] block->W_max;
    delete[] block->W_min;
    delete block;
}

void freeGSYN(SYNBlockVAR *gblock)
{
    SYNBlockVAR *tmp = toCPU(gblock, 1);
    gpuFree(tmp->src);
    gpuFree(tmp->tar);
    gpuFree(tmp->weight);
    gpuFree(tmp->delay);
    gpuFree(tmp->A_LTP);
    gpuFree(tmp->A_LTD);
    gpuFree(tmp->TAU_LTD);
    gpuFree(tmp->TAU_LTP);
    gpuFree(tmp->W_max);
    gpuFree(tmp->W_min);
    gpuFree(gblock);
}
