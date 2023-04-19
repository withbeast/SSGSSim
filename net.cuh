#include "mem.cuh"
#include "lif.cuh"
#include "syn.cuh"
struct Net{
    LIFBlock* neurons;
    SYNBlock* syns;
};
Net* initHNet(int num);
void freeHNet(Net* net);
void net2GPU(Net* hnet,Net* gnet,int num);
Net* initGNet();
void freeGNet(Net* gnet);