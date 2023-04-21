//
// Created by 15838 on 2023/3/7.
//
#pragma once
#ifndef SIMPLECPUSIM_MACRO_H
#define SIMPLECPUSIM_MACRO_H

#include <vector>
#include <queue>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <time.h>
#include <stdio.h>
#include <array>

#include <random>
const float TYPE_O2O=0;
const float TYPE_FULL=1;

typedef float real;
class Config
{
public:
    static real STEP;
    /// @brief 时间片长度 ms
    static real DT;
    /// @brief 权重变化量
    static real DW;
    /// @brief 设置时间片长度
    /// @param time_step 时间片时长 ms
    static void setTimestep(real time_step)
    {
        STEP = time_step;
        DT=time_step;
    }
    /// @brief 获取时间片数
    /// @param time 输入仿真时长 s
    /// @return 
    static int getSteps(real time){
        return round(time*1000/STEP);
    }
    static void setDW(real dw){
        DW=dw;
    }
};
real Config::STEP = 0.001;
real Config::DT=0.001;
real Config::DW=0.01;


struct LIFConst
{
    /// @brief Resting membrane potential 静息膜电位 mV
    real V_rest;
    /// @brief Reset membrane potential 重置膜电位 mV
    real V_reset;
    /// @brief Membrane capacitance 膜电容 nF
    real C_m;
    /// @brief Membrane time constant 膜时间常数 ms
    real Tau_m;
    /// @brief Refractory period 不应期 ms
    real Refrac_period;
    /// @brief Excitatory synaptic time constant 兴奋性突触时间常数 ms
    real Tau_exc;
    /// @brief Inhibitory synaptic time constant 抑制性突触时间常数 ms
    real Tau_inh;
    /// @brief Spike threshold 脉冲阈值 mV
    real V_thresh;
    /// @brief Injected current amplitude 注入电流幅值 nA
    real I_offset;

    /// 中间参数
    real _P22;
    real _P21exc;
    real _P21inh;
    real _P11exc;
    real _P11inh;
    /// @brief 不应期时间片数
    int Refrac_step;

    LIFConst()
    {
        V_rest = 0;          // mV
        V_reset = 0;         // mV
        C_m = 0.25;          // nF
        Tau_m = 10.0;        // ms
        Refrac_period = 4.0; // ms
        Tau_exc = 20;        // ms;
        Tau_inh = 30;        // ms
        V_thresh = 15;       // mV
        I_offset = 0;        // nA

        /// 中间参数
        real dt = Config::DT; // ms
        Refrac_step = Refrac_period / dt;
        _P22 = std::exp(-dt / Tau_m);
        _P11exc = std::exp(-dt / Tau_exc);
        _P11inh = std::exp(-dt / Tau_inh);
        _P21exc = Tau_m * Tau_exc / (C_m * 1000 * (Tau_exc - Tau_m));
        _P21inh = Tau_m * Tau_inh / (C_m * 1000 * (Tau_inh - Tau_m));
    }
};
struct STDPConst{
    real A_LTP;
    real A_LTD;
    real TAU_LTP;
    real TAU_LTD;
    real W_max;
    real W_min;
    STDPConst(){
        A_LTP=0.1;
        A_LTD=-0.01;
        TAU_LTP=17;
        TAU_LTD=34;
        W_max=40;
        W_min=0;
    }
};

struct LIFNeuron
{
public:
    /// @brief 静态神经元参数
    LIFConst *Consts;
    /// @brief 神经元id
    int id;
    /// @brief 是否激活
    bool fired;
    /// @brief 激活次数
    int firecnt;
    /// @brief 最近激活时间
    int last_fired;
    /// @brief 突触输入电流
    real I_syn_exc;
    real I_syn_inh;
    /// @brief 神经元电流
    real I_exc;
    real I_inh;
    /// @brief 神经元膜电位
    real Vm;
    /// @brief 不应期状态值
    int refrac_state;
    bool source;
    real rate;

public:
    LIFNeuron(int _id, bool _isSource,LIFConst* consts):id(_id),source(_isSource),fired(false),firecnt(0),last_fired(0)
    {
        Vm = 0;
        I_syn_exc = 0;
        I_syn_inh = 0;
        I_exc = 0;
        I_inh = 0;
        Consts = consts;
        refrac_state = 0;
    }
};
struct Synapse
{
    public:
    int src;
    int tar;
    real weight;
    real delay;
};

#endif // SIMPLECPUSIM_MACRO_H
