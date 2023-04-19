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
enum NeuronType
{
    RAW,
    LIF,
    POISSON,
    LIF2
};
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
#endif // SIMPLECPUSIM_MACRO_H
