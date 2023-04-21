#include "net.cuh"

Net *copyNet2GPU(Net *cnet)
{
    NEUBlock *gneurons = copyLIF2GPU(cnet->neurons);
    SYNBlock *gsyns = copySYN2GPU(cnet->syns);
    Net *tmp = new Net;
    tmp->neurons = gneurons;
    tmp->syns = gsyns;
    Net *gnet = toGPU(tmp, 1);
    delete tmp;
    return gnet;
}
void freeCNet(Net *cnet)
{
    freeLIFData(cnet->neurons);
    freeSYNData(cnet->syns);
    delete cnet;
}
void freeGNet(Net *gnet)
{
    Net *tmp = toCPU(gnet, 1);
    freeGLIF(tmp->neurons);
    freeGSYN(tmp->syns);
    gpuFree(gnet);
}



NetVAR *copyNet2GPU(NetVAR *cnet)
{
    NEUBlockVAR *gneurons = copyLIF2GPU(cnet->neurons);
    SYNBlockVAR *gsyns = copySYN2GPU(cnet->syns);
    NetVAR *tmp = new NetVAR;
    tmp->neurons = gneurons;
    tmp->syns = gsyns;
    NetVAR *gnet = toGPU(tmp, 1);
    delete tmp;
    return gnet;
}
void freeCNet(NetVAR *cnet)
{
    freeLIFData(cnet->neurons);
    freeSYNData(cnet->syns);
    delete cnet;
}
void freeGNet(NetVAR *gnet)
{
    NetVAR *tmp = toCPU(gnet, 1);
    freeGLIF(tmp->neurons);
    freeGSYN(tmp->syns);
    gpuFree(gnet);
}