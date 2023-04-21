
#include "netgen.hpp"
#include "network.hpp"
#include "model.hpp"
#include "time.h"
#include <iostream>
#include "sim.cu"
#include <fstream>
real random0001()
{
    return rand() % (999 + 1) / (float)(999 + 1);
}
Net *buildNetwork(Network &net, int steps)
{
    NEUBlock *neus = initLIFData(net.neurons.size(), steps);
    SYNBlock *syns = initSYNData(net.synapses.size());
    for (int i = 0; i < net.neurons.size(); i++)
    {
        if (net.neurons[i]->source)
        {
            neus->source[i] = true;
            neus->rate[i] = random0001();
            for (int j = 0; j < steps; j++)
            {
                neus->rand[i * steps + j] = random0001();
            }
        }
        else
        {
            neus->source[i] = false;
        }
    }
    syns->num = net.synapses.size();
    for (int i = 0; i < net.synapses.size(); i++)
    {
        syns->src[i] = net.synapses[i]->src;
        syns->tar[i] = net.synapses[i]->tar;
        syns->delay[i] = net.synapses[i]->delay / Config::STEP;
        syns->weight[i] = net.synapses[i]->weight;
    }
    Net *cnet = new Net;
    cnet->neurons = neus;
    cnet->syns = syns;
    return cnet;
}
NetVAR *buildNetworkVAR(Network &net, int steps)
{
    NEUBlockVAR *neus = initLIFDataVAR(net.neurons.size(), steps);
    SYNBlockVAR *syns = initSYNDataVAR(net.synapses.size());
    for (int i = 0; i < net.neurons.size(); i++)
    {
        if (net.neurons[i]->source)
        {
            neus->source[i] = true;
            neus->rate[i] = random0001();
            for (int j = 0; j < steps; j++)
            {
                neus->rand[i * steps + j] = random0001();
            }
        }
        else
        {
            neus->source[i] = false;
        }
    }
    syns->num = net.synapses.size();
    for (int i = 0; i < net.synapses.size(); i++)
    {
        syns->src[i] = net.synapses[i]->src;
        syns->tar[i] = net.synapses[i]->tar;
        syns->delay[i] = net.synapses[i]->delay / Config::STEP;
        syns->weight[i] = net.synapses[i]->weight;
    }
    NetVAR *cnet = new NetVAR;
    cnet->neurons = neus;
    cnet->syns = syns;
    return cnet;
}

int main(void)
{
    srand((int)time(0));
    std::vector<int> xarr = {10'000, 20'000, 30'000, 40'000, 50'000, 60'000, 70'000, 80'000, 90'000};
    std::vector<int> varr(xarr.size());
    std::vector<int> carr(xarr.size());
    for (int i = 0; i < xarr.size(); i++)
    {
        int steps = xarr[i];
        Model model;
        LIFConst *consts = new LIFConst;
        auto p1 = model.createPop(100, true, consts);
        auto p2 = model.createPop(100, false, consts);
        auto p3 = model.createPop(100, false, consts);
        model.connect(p1, p2, {1, 2}, {0.002, 0.008}, 0.5);
        model.connect(p2, p3, {1, 2}, {0.002, 0.008}, 0.5);
        model.connect(p1, p3, {1, 2}, {0.002, 0.008}, 0.5);
        Network net = NetGen::genNet(&model);
        Net *cnet = buildNetwork(net, steps);
        NetVAR *cnetv = buildNetworkVAR(net, steps);
        time_t start1, end1,start2,end2;
        start1 = clock();
        simulate1(cnet, steps);
        end1 = clock();
        start2 = clock();
        simulate2(cnetv, steps);
        end2 = clock();
        int celapse=(end1 - start1) / 1000;
        int velapse=(end2 - start2) / 1000;
        printf("const elapse time:%d ms\n", celapse);
        printf("var elapse time:%d ms\n", velapse);
        varr[i] = velapse;
        carr[i] = celapse;
    }
    std::fstream file;
    std::string name="pop3_100";
    file.open("../"+name+".txt",std::ios::out);
    if (file.is_open())
    {
        for (int i = 0; i < xarr.size(); i++)
        {
            file<<xarr[i]<<" ";
        }
        file<<std::endl;
        for (int i = 0; i < xarr.size(); i++)
        {
            file<<carr[i]<<" ";
        }
        file<<std::endl;
        for (int i = 0; i < xarr.size(); i++)
        {
            file<<varr[i]<<" ";
        }
    }
    file.close();
    return 0;
}