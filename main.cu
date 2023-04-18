#include "lif.cuh"
inline void CheckCall(cudaError_t err, const char *file, int line)
{
    const cudaError_t error = err;
    if (error != cudaSuccess)
    {
        printf("Error:%s.Line %d,", file, line);
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
#define CUDACHECK(x) CheckCall(x, __FILE__, __LINE__)

LIFBlock* initData(int num){
    LIFBlock* block=new LIFBlock;
    block->V_m=new real[num]();
    block->Fired=new bool[num]();
    block->FireCnt=new int[num]();
    block->I_exc=new real[num]();
    block->I_inh=new real[num]();
    block->I_syn_exc=new real[num]();
    block->I_syn_inh=new real[num]();
    block->Refrac_state=new int[num]();
    block->n=num;
    return block;
}
void freeData(LIFBlock* block){
    delete[] block->V_m;
    delete[] block->FireCnt;
    delete[] block->Fired;
    delete[] block->I_exc;
    delete[] block->I_inh;
    delete[] block->I_syn_exc;
    delete[] block->I_syn_inh;
    delete[] block->Refrac_state;
    delete block;
}
template<typename T>
T* toGPU(T* cpu, int size)
{
    T * ret;
    CUDACHECK(cudaMalloc((void**)&(ret), sizeof(T) * size));
    CUDACHECK(cudaMemcpy(ret, cpu, sizeof(T)*size, cudaMemcpyHostToDevice));
    return ret;
}


LIFBlock* copy2GPU(LIFBlock* cblock){
    LIFBlock* gblock=nullptr;
    LIFBlock* tmp=new LIFBlock;
    tmp->n=cblock->n;
    tmp->V_m=toGPU(cblock->V_m,cblock->n);
    tmp->I_exc=toGPU(cblock->I_exc,cblock->n);
    tmp->I_inh=toGPU(cblock->I_inh,cblock->n);
    tmp->I_syn_exc=toGPU(cblock->I_syn_exc,cblock->n);
    tmp->I_syn_inh=toGPU(cblock->I_syn_inh,cblock->n);
    tmp->Fired=toGPU(cblock->Fired,cblock->n);
    tmp->FireCnt=toGPU(cblock->FireCnt,cblock->n);
    tmp->Refrac_state=toGPU(cblock->Refrac_state,cblock->n);
    gblock=toGPU(tmp,1);
    delete tmp;
    return gblock;
}

__global__ void simulate(LIFBlock* block)
{
    int id =  blockIdx.x * blockDim.x + threadIdx.x;
    simulateLIF(block,id);
    // printf("Hello World from GPU!\n");
    printf("0'vm:%f\n",block->V_m[0]);
}

int main(void)
{
    LIFBlock* block=initData(10);
    LIFBlock* gblock=copy2GPU(block);

    // Hello from gpu
    // simulate<<<1, 10>>>(gblock);
    std::cout<<"simulate complete"<<std::endl;

    cudaDeviceReset();
    freeData(block);
    return 0;
}