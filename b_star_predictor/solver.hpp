#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// 1. B Steps Wala CUDA WHILE Graph Banana aur Chalana
float run_while_graph(int B,float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,int* d_steps,int* out_steps);

// 2. Micro-Probe: 1 Step (t_step) aur 1 Check (t_check) ka time naapna
void probe_timings(float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,int* d_steps,float* out_t_step_us,float* out_t_check_us);

// 3. Baseline 1: Normal CPU-Driven While Loop chalana
float run_cpu_loop(float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,int* out_steps);
