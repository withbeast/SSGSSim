#pragma once
#include "mem.cuh"
#include "lif.cuh"
#include "syn.cuh"

struct Net
{
    NEUBlock *neurons;
    SYNBlock *syns;
};
struct NetVAR
{
    NEUBlockVAR *neurons;
    SYNBlockVAR *syns;
};
Net *copyNet2GPU(Net *cnet);
void freeCNet(Net *cnet);
void freeGNet(Net *gnet);
NetVAR *copyNet2GPU(NetVAR *cnet);
void freeCNet(NetVAR *cnet);
void freeGNet(NetVAR *gnet);
