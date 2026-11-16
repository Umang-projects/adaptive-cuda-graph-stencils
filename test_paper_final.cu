#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>

// 1A. 5-Point Stencil Math Kernel (Jacobi)
__global__ void heat_step_kernel(const float* In,float* Out,int N){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<=0 || x>=N-1 || y<=0 || y>=N-1) return;
    int idx=y*N+x;
    Out[idx]=0.25f*(In[idx-1]+In[idx+1]+In[idx-N]+In[idx+N]);
}

// 1B. NAYA: 9-Point Moore Stencil Kernel (Orthogonal + Diagonal Neighbors)
__global__ void heat_9pt_kernel(const float* In,float* Out,int N){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<=0 || x>=N-1 || y<=0 || y>=N-1) return;
    int idx=y*N+x;
    float ortho=In[idx-1]+In[idx+1]+In[idx-N]+In[idx+N];
    float diag=In[idx-N-1]+In[idx-N+1]+In[idx+N-1]+In[idx+N+1];
    Out[idx]=0.16f*ortho+0.09f*diag;
}

// 2. Convergence Check Kernel
__global__ void check_error_kernel(const float* A,const float* B,int N,float tol,int* d_flag){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<=0 || x>=N-1 || y<=0 || y>=N-1) return;
    int idx=y*N+x;
    if(fabsf(A[idx]-B[idx])>tol){
        *d_flag=1;
    }
}

// 3. Fused Set-Signal + Reset-Flag Kernel (For WHILE Graph)
__global__ void set_and_reset_signal_kernel(cudaGraphConditionalHandle handle,int* d_flag,int* d_steps,int B){
    cudaGraphSetConditional(handle,*d_flag);
    *d_flag=0;
    *d_steps+=B;
}

// Helper: Launch 1 step of chosen stencil (mode 0 = 5-pt, mode 1 = 9-pt)
inline void launch_stencil(int mode,dim3 grid,dim3 block,cudaStream_t stream,const float* In,float* Out,int N){
    if(mode==0) heat_step_kernel<<<grid,block,0,stream>>>(In,Out,N);
    else heat_9pt_kernel<<<grid,block,0,stream>>>(In,Out,N);
}

// Builder A: Static Math Graph (with optional Build Time measurement)
cudaGraphExec_t build_static_graph(int steps,float* d_A,float* d_B,int N,cudaStream_t stream,int mode=0,float* out_build_ms=nullptr){
    auto t0=std::chrono::high_resolution_clock::now();
    cudaGraph_t graph;
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
    for(int i=0;i<steps/2;i++){
        launch_stencil(mode,grid,block,stream,d_A,d_B,N);
        launch_stencil(mode,grid,block,stream,d_B,d_A,N);
    }
    cudaStreamEndCapture(stream,&graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec,graph,0);
    cudaGraphDestroy(graph);
    auto t1=std::chrono::high_resolution_clock::now();
    if(out_build_ms) *out_build_ms=std::chrono::duration<float,std::milli>(t1-t0).count();
    return exec;
}

// Builder B: Device-Side Conditional WHILE Graph (Zero CPU Sync!)
cudaGraphExec_t build_while_graph(int B,float* d_A,float* d_B,int N,float tol,int* d_flag,int* d_steps,cudaStream_t stream,cudaGraph_t* out_outer,int mode=0,float* out_build_ms=nullptr){
    auto t0=std::chrono::high_resolution_clock::now();
    cudaGraphCreate(out_outer,0);
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle,*out_outer,1,cudaGraphCondAssignDefault);

    cudaGraphNodeParams params={};
    params.type=cudaGraphNodeTypeConditional;
    params.conditional.handle=handle;
    params.conditional.type=cudaGraphCondTypeWhile;
    params.conditional.size=1;

    cudaGraphNode_t while_node;
    cudaGraphAddNode(&while_node,*out_outer,nullptr,0,&params);
    cudaGraph_t body=params.conditional.phGraph_out[0];

    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaStreamBeginCaptureToGraph(stream,body,nullptr,nullptr,0,cudaStreamCaptureModeGlobal);
    for(int i=0;i<B/2;i++){
        launch_stencil(mode,grid,block,stream,d_A,d_B,N);
        launch_stencil(mode,grid,block,stream,d_B,d_A,N);
    }
    check_error_kernel<<<grid,block,0,stream>>>(d_A,d_B,N,tol,d_flag);
    set_and_reset_signal_kernel<<<1,1,0,stream>>>(handle,d_flag,d_steps,B);
    cudaStreamEndCapture(stream,&body);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec,*out_outer,0);
    auto t1=std::chrono::high_resolution_clock::now();
    if(out_build_ms) *out_build_ms=std::chrono::duration<float,std::milli>(t1-t0).count();
    return exec;
}

// Practical I-Cache Autotuner — Finds Best K* in {16, 32, 64, 128}
int autotune_k_cache(float* d_A,float* d_B,int N,cudaStream_t stream,cudaEvent_t ev_s,cudaEvent_t ev_e,int mode=0){
    int cands[4]={16,32,64,128};
    int best_K=128;
    float best_ms=1e9f;

    for(int i=0;i<4;i++){
        int K=cands[i];
        int reps=128/K;
        cudaGraphExec_t exec=build_static_graph(K,d_A,d_B,N,stream,mode);

        for(int w=0;w<20;w++){
            for(int r=0;r<reps;r++) cudaGraphLaunch(exec,stream);
        }
        cudaStreamSynchronize(stream);

        cudaEventRecord(ev_s,stream);
        for(int t=0;t<50;t++){
            for(int r=0;r<reps;r++) cudaGraphLaunch(exec,stream);
        }
        cudaEventRecord(ev_e,stream);
        cudaStreamSynchronize(stream);

        float ms=0.0f;
        cudaEventElapsedTime(&ms,ev_s,ev_e);
        if(ms<best_ms){
            best_ms=ms;
            best_K=K;
        }
        cudaGraphExecDestroy(exec);
    }
    return best_K;
}

// Run 1: Normal CPU Loop (No Graph)
float run_baseline_cpu(float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,cudaStream_t stream,cudaEvent_t ev_s,cudaEvent_t ev_e,int* out_steps,int mode=0){
    int bytes=N*N*sizeof(float);
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaMemcpy(d_A,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_init,bytes,cudaMemcpyHostToDevice);
    int h_flag=1,steps=0;

    cudaEventRecord(ev_s,stream);
    while(h_flag==1){
        launch_stencil(mode,grid,block,stream,d_A,d_B,N);
        launch_stencil(mode,grid,block,stream,d_B,d_A,N);
        steps+=2;
        cudaMemsetAsync(d_flag,0,sizeof(int),stream);
        check_error_kernel<<<grid,block,0,stream>>>(d_A,d_B,N,tol,d_flag);
        cudaMemcpyAsync(&h_flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(ev_e,stream);
    cudaStreamSynchronize(stream);
    float ms=0.0f;
    cudaEventElapsedTime(&ms,ev_s,ev_e);
    if(out_steps) *out_steps=steps;
    return ms;
}

// Run 2: Ekelund Static Graph (1 Launch of B per CPU Check)
float run_ekelund(cudaGraphExec_t exec_B,float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,cudaStream_t stream,cudaEvent_t ev_s,cudaEvent_t ev_e){
    int bytes=N*N*sizeof(float);
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaMemcpy(d_A,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_init,bytes,cudaMemcpyHostToDevice);
    int h_flag=1;

    cudaEventRecord(ev_s,stream);
    while(h_flag==1){
        cudaGraphLaunch(exec_B,stream);
        cudaMemsetAsync(d_flag,0,sizeof(int),stream);
        check_error_kernel<<<grid,block,0,stream>>>(d_A,d_B,N,tol,d_flag);
        cudaMemcpyAsync(&h_flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(ev_e,stream);
    cudaStreamSynchronize(stream);
    float ms=0.0f;
    cudaEventElapsedTime(&ms,ev_s,ev_e);
    return ms;
}

// Run 3: Autotuned Cache-Pipelined Graph (Launches exec_kstar 'reps' times per CPU Check)
float run_pipelined_kstar(cudaGraphExec_t exec_kstar,int reps,float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,cudaStream_t stream,cudaEvent_t ev_s,cudaEvent_t ev_e){
    int bytes=N*N*sizeof(float);
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);
    cudaMemcpy(d_A,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_init,bytes,cudaMemcpyHostToDevice);
    int h_flag=1;

    cudaEventRecord(ev_s,stream);
    while(h_flag==1){
        for(int r=0;r<reps;r++){
            cudaGraphLaunch(exec_kstar,stream);
        }
        cudaMemsetAsync(d_flag,0,sizeof(int),stream);
        check_error_kernel<<<grid,block,0,stream>>>(d_A,d_B,N,tol,d_flag);
        cudaMemcpyAsync(&h_flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(ev_e,stream);
    cudaStreamSynchronize(stream);
    float ms=0.0f;
    cudaEventElapsedTime(&ms,ev_s,ev_e);
    return ms;
}

// Run 4: Our Conditional WHILE Graph (1 Single Launch, ZERO CPU Sync!)
float run_while_method(cudaGraphExec_t exec_while,float* d_A,float* d_B,const float* h_init,int N,int* d_flag,int* d_steps,cudaStream_t stream,cudaEvent_t ev_s,cudaEvent_t ev_e,int* out_steps){
    int bytes=N*N*sizeof(float);
    cudaMemcpy(d_A,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemsetAsync(d_flag,0,sizeof(int),stream);
    cudaMemsetAsync(d_steps,0,sizeof(int),stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(ev_s,stream);
    cudaGraphLaunch(exec_while,stream);
    cudaEventRecord(ev_e,stream);
    cudaStreamSynchronize(stream);

    float ms=0.0f;
    cudaEventElapsedTime(&ms,ev_s,ev_e);
    if(out_steps) cudaMemcpy(out_steps,d_steps,sizeof(int),cudaMemcpyDeviceToHost);
    return ms;
}

// Helper: Runs Multi-Grid Showdown Table for a given Stencil Mode (0 = 5-Point, 1 = 9-Point)
void run_grid_table(const char* title,int mode,const char* gpu_name,int* d_flag,int* d_steps,cudaStream_t stream,cudaEvent_t ev_s,cudaEvent_t ev_e){
    printf("\n===================================================================================================================\n");
    printf("  %s [%s]: FAIR COMPARISON WITH AUTOTUNED I-CACHE K* (100 Warmups, 10-Run Average)\n",title,gpu_name);
    printf("===================================================================================================================\n");
    printf(" Grid Size | 1.NoGraph | 2A.Ekelund | 2B.Our WHILE | WHILE Gain | Picked K* | 3A.Ekelund  | 3B.Autotune | I-Cache\n");
    printf("           | Baseline  | (B=16,CPU) | (B=16,0Sync) | (vs 2A)    | (for 128) | (Single128) | (K* x Reps) | Gain(3A)\n");
    printf("-----------+-----------+------------+--------------+------------+-----------+-------------+-------------+----------\n");

    int grids[5]={32,64,128,256,512};
    for(int g=0;g<5;g++){
        int N=grids[g];
        int bytes=N*N*sizeof(float);
        float tol=0.02f;
        float* h_init=new float[N*N]();
        for(int x=0;x<N;x++){ h_init[x]=100.0f; }

        float *d_A,*d_B;
        cudaMalloc(&d_A,bytes); cudaMalloc(&d_B,bytes);

        int K_star=autotune_k_cache(d_A,d_B,N,stream,ev_s,ev_e,mode);
        int reps=128/K_star;

        cudaGraphExec_t exec_16   =build_static_graph(16,d_A,d_B,N,stream,mode);
        cudaGraphExec_t exec_kstar=build_static_graph(K_star,d_A,d_B,N,stream,mode);
        cudaGraphExec_t exec_128  =build_static_graph(128,d_A,d_B,N,stream,mode);
        cudaGraph_t outer_w;
        cudaGraphExec_t exec_wh16 =build_while_graph(16,d_A,d_B,N,tol,d_flag,d_steps,stream,&outer_w,mode);

        for(int w=0;w<100;w++){ cudaGraphLaunch(exec_kstar,stream); }
        cudaStreamSynchronize(stream);

        float t_base=run_baseline_cpu(d_A,d_B,h_init,N,tol,d_flag,stream,ev_s,ev_e,nullptr,mode);

        float sum_ek16=0.0f,sum_wh16=0.0f,sum_ek128=0.0f,sum_pipe=0.0f;
        for(int r=0;r<10;r++){
            sum_ek16 +=run_ekelund(exec_16,d_A,d_B,h_init,N,tol,d_flag,stream,ev_s,ev_e);
            sum_wh16 +=run_while_method(exec_wh16,d_A,d_B,h_init,N,d_flag,d_steps,stream,ev_s,ev_e,nullptr);
            sum_ek128+=run_ekelund(exec_128,d_A,d_B,h_init,N,tol,d_flag,stream,ev_s,ev_e);
            sum_pipe +=run_pipelined_kstar(exec_kstar,reps,d_A,d_B,h_init,N,tol,d_flag,stream,ev_s,ev_e);
        }
        float t_ek16 =sum_ek16/10.0f;
        float t_wh16 =sum_wh16/10.0f;
        float t_ek128=sum_ek128/10.0f;
        float t_pipe =sum_pipe/10.0f;

        float gain_while=((t_ek16-t_wh16)/t_ek16)*100.0f;
        float gain_cache=((t_ek128-t_pipe)/t_ek128)*100.0f;

        char k_str[16];
        snprintf(k_str,sizeof(k_str),"%dx%d",K_star,reps);

        printf(" %4dx%-4d | %7.3fms |  %7.3fms |    %7.3fms |   %+5.1f%%  | %9s |   %7.3fms |   %7.3fms |  %+5.1f%%\n",
               N,N,t_base,t_ek16,t_wh16,gain_while,k_str,t_ek128,t_pipe,gain_cache);

        cudaGraphExecDestroy(exec_16);
        cudaGraphExecDestroy(exec_kstar);
        cudaGraphExecDestroy(exec_128);
        cudaGraphExecDestroy(exec_wh16);
        cudaGraphDestroy(outer_w);
        delete[] h_init; cudaFree(d_A); cudaFree(d_B);
    }
    printf("===================================================================================================================\n");
}

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    printf("\n>>> RUNNING HARDWARE-ADAPTIVE BENCHMARK ON: %s (%d SMs) <<<\n",prop.name,prop.multiProcessorCount);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    cudaEvent_t ev_s,ev_e;
    cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);

    int *d_flag,*d_steps;
    cudaMalloc(&d_flag,sizeof(int));
    cudaMalloc(&d_steps,sizeof(int));

    // GLOBAL GPU CLOCK LOCK WARMUP (1000 Kernels)
    {
        float *wA,*wB;
        cudaMalloc(&wA,256*256*sizeof(float));
        cudaMalloc(&wB,256*256*sizeof(float));
        dim3 b(16,16),g(16,16);
        for(int w=0;w<1000;w++){ heat_step_kernel<<<g,b,0,stream>>>(wA,wB,256); }
        cudaStreamSynchronize(stream);
        cudaFree(wA); cudaFree(wB);
    }

    // =========================================================================
    // TABLE 1: B SWEEP (2 to 256) ON 256x256 — EKELUND vs OUR 'WHILE' + BUILD TIME
    // =========================================================================
    {
        int N=256,bytes=N*N*sizeof(float);
        float tol=0.02f;
        float* h_init=new float[N*N]();
        for(int x=0;x<N;x++){ h_init[x]=100.0f; }
        float *d_A,*d_B;
        cudaMalloc(&d_A,bytes); cudaMalloc(&d_B,bytes);

        int true_steps=0;
        float t_cpu=run_baseline_cpu(d_A,d_B,h_init,N,tol,d_flag,stream,ev_s,ev_e,&true_steps,0);

        printf("\n========================================================================================================\n");
        printf("  TABLE 1 [%s]: BATCH SWEEP ON 256x256 (True Steps=%d | CPU Base=%.3f ms)\n",prop.name,true_steps,t_cpu);
        printf("========================================================================================================\n");
        printf("   B   | Steps Ran | Build T_C (ms) | Ekelund Static (CPU Sync) | Our WHILE Graph (0 Sync) | Speedup\n");
        printf("-------+-----------+----------------+---------------------------+--------------------------+----------\n");

        int B_list[8]={2,4,8,16,32,64,128,256};
        for(int i=0;i<8;i++){
            int B=B_list[i];
            float build_ms=0.0f;
            cudaGraphExec_t exec_ek=build_static_graph(B,d_A,d_B,N,stream,0,&build_ms);
            cudaGraph_t outer_w;
            cudaGraphExec_t exec_wh=build_while_graph(B,d_A,d_B,N,tol,d_flag,d_steps,stream,&outer_w,0);

            for(int w=0;w<100;w++){ cudaGraphLaunch(exec_ek,stream); }
            cudaStreamSynchronize(stream);

            float sum_ek=0.0f,sum_wh=0.0f;
            int steps_ran=0;
            for(int r=0;r<10;r++){
                sum_ek+=run_ekelund(exec_ek,d_A,d_B,h_init,N,tol,d_flag,stream,ev_s,ev_e);
                sum_wh+=run_while_method(exec_wh,d_A,d_B,h_init,N,d_flag,d_steps,stream,ev_s,ev_e,&steps_ran);
            }
            float t_ek=sum_ek/10.0f;
            float t_wh=sum_wh/10.0f;
            float gain=((t_ek-t_wh)/t_ek)*100.0f;

            printf("  %3d  |   %5d   |   %8.3f ms  |          %8.3f ms       |         %8.3f ms       | %+5.1f%%\n",
                   B,steps_ran,build_ms,t_ek,t_wh,gain);

            cudaGraphExecDestroy(exec_ek);
            cudaGraphExecDestroy(exec_wh);
            cudaGraphDestroy(outer_w);
        }
        printf("========================================================================================================\n");
        delete[] h_init; cudaFree(d_A); cudaFree(d_B);
    }

    // =========================================================================
    // TABLE 2: 5-POINT JACOBI STENCIL MULTI-GRID SHOWDOWN
    // =========================================================================
    run_grid_table("TABLE 2 (5-POINT JACOBI)",0,prop.name,d_flag,d_steps,stream,ev_s,ev_e);

    // =========================================================================
    // TABLE 3: PURE MICRO-ARCHITECTURAL I-CACHE / DESCRIPTOR SWEEP (256 STEPS)
    // =========================================================================
    {
        int N=128,bytes=N*N*sizeof(float);
        float *d_A,*d_B;
        cudaMalloc(&d_A,bytes); cudaMalloc(&d_B,bytes);

        printf("\n=====================================================================================\n");
        printf("  TABLE 3 [%s]: PURE I-CACHE / DESCRIPTOR SWEEP (128x128 Grid, Total=256 Steps)\n",prop.name);
        printf("=====================================================================================\n");
        printf("  Kernels/Graph (K) | Replays | Total Steps | Build T_C (ms) | Exec T_E (ms) | VRAM\n");
        printf("--------------------+---------+-------------+----------------+---------------+-------\n");

        int K_list[8]={2,4,8,16,32,64,128,256};
        for(int i=0;i<8;i++){
            int K=K_list[i];
            int reps=256/K;
            size_t free_b=0,tot=0,free_a=0;
            cudaMemGetInfo(&free_b,&tot);

            float build_ms=0.0f;
            cudaGraphExec_t exec=build_static_graph(K,d_A,d_B,N,stream,0,&build_ms);

            cudaMemGetInfo(&free_a,&tot);
            float vram_mb=(free_b>=free_a)?((float)(free_b-free_a)/(1024.0f*1024.0f)):0.0f;

            for(int w=0;w<20;w++){
                for(int r=0;r<reps;r++) cudaGraphLaunch(exec,stream);
            }
            cudaStreamSynchronize(stream);

            cudaEventRecord(ev_s,stream);
            for(int t=0;t<50;t++){
                for(int r=0;r<reps;r++) cudaGraphLaunch(exec,stream);
            }
            cudaEventRecord(ev_e,stream);
            cudaStreamSynchronize(stream);

            float ms=0.0f;
            cudaEventElapsedTime(&ms,ev_s,ev_e);
            ms/=50.0f;

            printf("         %4d       |   %4d  |     %4d    |    %8.3f ms |   %8.4f ms | %4.1fMB\n",
                   K,reps,K*reps,build_ms,ms,vram_mb);

            cudaGraphExecDestroy(exec);
        }
        printf("=====================================================================================\n");
        cudaFree(d_A); cudaFree(d_B);
    }

    // =========================================================================
    // TABLE 4: 9-POINT MOORE STENCIL MULTI-GRID SHOWDOWN (2nd Benchmark!)
    // =========================================================================
    run_grid_table("TABLE 4 (9-POINT MOORE)",1,prop.name,d_flag,d_steps,stream,ev_s,ev_e);

    cudaFree(d_flag); cudaFree(d_steps);
    cudaStreamDestroy(stream);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e);
    return 0;
}
