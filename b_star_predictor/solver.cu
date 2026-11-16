#include "solver.hpp"

// KADAM 1..B: 1 Step Garmi Phailana (Read In -> Write Out)
__global__ void heat_step_kernel(const float* In,float* Out,int N){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<=0 || x>=N-1 || y<=0 || y>=N-1) return;

    int idx=y*N+x;
    Out[idx]=0.25f*(In[idx-1]+In[idx+1]+In[idx-N]+In[idx+N]);
}

// KADAM 5: Jhanda Reset Karo (0) aur Steps Gino (+B)
__global__ void reset_flag_kernel(int* d_keep_going,int* d_steps,int B){
    *d_keep_going=0;
    *d_steps+=B;
}

// KADAM 6: Chakhne Wala Kernel (Agar koi dabba > tol badla toh Jhanda = 1)
__global__ void check_error_kernel(const float* A,const float* B,int N,float tol,int* d_keep_going){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<=0 || x>=N-1 || y<=0 || y>=N-1) return;

    int idx=y*N+x;
    if(fabsf(A[idx]-B[idx])>tol){
        *d_keep_going=1;
    }
}

// KADAM 7: Traffic Signal (handle) ko 1 (Green) ya 0 (Red) set karo
__global__ void set_signal_kernel(cudaGraphConditionalHandle handle,const int* d_keep_going){
    cudaGraphSetConditional(handle,*d_keep_going);
}

// FUNCTION 1: Tera exact run_while_graph
float run_while_graph(int B,float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,int* d_steps,int* out_steps){
    int bytes=N*N*sizeof(float);
    cudaMemcpy(d_A,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemset(d_steps,0,sizeof(int));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);

    cudaGraph_t outer_graph;
    cudaGraphCreate(&outer_graph,0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle,outer_graph,1,cudaGraphCondAssignDefault);

    cudaGraphNodeParams params={};
    params.type=cudaGraphNodeTypeConditional;
    params.conditional.handle=handle;
    params.conditional.type=cudaGraphCondTypeWhile;
    params.conditional.size=1;

    cudaGraphNode_t while_node;
    cudaGraphAddNode(&while_node,outer_graph,nullptr,0,&params);
    cudaGraph_t body_graph=params.conditional.phGraph_out[0];

    dim3 block(16,16);
    dim3 grid((N+15)/16,(N+15)/16);

    cudaStreamBeginCaptureToGraph(stream,body_graph,nullptr,nullptr,0,cudaStreamCaptureModeGlobal);
    for(int i=0;i<B/2;i++){
        heat_step_kernel<<<grid,block,0,stream>>>(d_A,d_B,N);
        heat_step_kernel<<<grid,block,0,stream>>>(d_B,d_A,N);
    }
    reset_flag_kernel<<<1,1,0,stream>>>(d_flag,d_steps,B);
    check_error_kernel<<<grid,block,0,stream>>>(d_A,d_B,N,tol,d_flag);
    set_signal_kernel<<<1,1,0,stream>>>(handle,d_flag);
    cudaStreamEndCapture(stream,&body_graph);

    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec,outer_graph,0);

    cudaEvent_t ev_s,ev_e;
    cudaEventCreate(&ev_s);
    cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s,stream);
    cudaGraphLaunch(graph_exec,stream);
    cudaEventRecord(ev_e,stream);
    cudaStreamSynchronize(stream);

    float ms=0.0f;
    cudaEventElapsedTime(&ms,ev_s,ev_e);
    cudaMemcpy(out_steps,d_steps,sizeof(int),cudaMemcpyDeviceToHost);

    cudaEventDestroy(ev_s);
    cudaEventDestroy(ev_e);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(outer_graph);
    cudaStreamDestroy(stream);
    return ms;
}

// FUNCTION 2: Tera exact Micro-Probe code (main se utha kar yahan rakha kyunki isme <<<...>>> hai)
void probe_timings(float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,int* d_steps,float* out_t_step_us,float* out_t_check_us){
    int bytes=N*N*sizeof(float);
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);

    cudaEvent_t ev_s,ev_e;
    cudaEventCreate(&ev_s);
    cudaEventCreate(&ev_e);

    cudaMemcpy(d_A,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_init,bytes,cudaMemcpyHostToDevice);
    for(int w=0;w<20;w++){ heat_step_kernel<<<grid,block>>>(d_A,d_B,N); }
    cudaDeviceSynchronize();

    cudaEventRecord(ev_s);
    for(int r=0;r<200;r++){ heat_step_kernel<<<grid,block>>>(d_A,d_B,N); }
    cudaEventRecord(ev_e);
    cudaEventSynchronize(ev_e);
    float step_ms=0.0f;
    cudaEventElapsedTime(&step_ms,ev_s,ev_e);
    *out_t_step_us=(step_ms*1000.0f)/200.0f;

    cudaEventRecord(ev_s);
    for(int r=0;r<200;r++){
        reset_flag_kernel<<<1,1>>>(d_flag,d_steps,2);
        check_error_kernel<<<grid,block>>>(d_A,d_B,N,tol,d_flag);
    }
    cudaEventRecord(ev_e);
    cudaEventSynchronize(ev_e);
    float check_ms=0.0f;
    cudaEventElapsedTime(&check_ms,ev_s,ev_e);
    *out_t_check_us=(check_ms*1000.0f)/200.0f;

    cudaEventDestroy(ev_s);
    cudaEventDestroy(ev_e);
}

// FUNCTION 3: Tera exact Baseline 1 CPU Loop (main se utha kar yahan rakha kyunki isme <<<...>>> hai)
float run_cpu_loop(float* d_A,float* d_B,const float* h_init,int N,float tol,int* d_flag,int* out_steps){
    int bytes=N*N*sizeof(float);
    dim3 block(16,16),grid((N+15)/16,(N+15)/16);

    cudaMemcpy(d_A,h_init,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_init,bytes,cudaMemcpyHostToDevice);
    int h_keep_going=1,N_conv=0;

    cudaEvent_t ev_s,ev_e;
    cudaEventCreate(&ev_s);
    cudaEventCreate(&ev_e);

    cudaEventRecord(ev_s);
    while(h_keep_going==1){
        heat_step_kernel<<<grid,block>>>(d_A,d_B,N);
        heat_step_kernel<<<grid,block>>>(d_B,d_A,N);
        N_conv+=2;
        cudaMemsetAsync(d_flag,0,sizeof(int));
        check_error_kernel<<<grid,block>>>(d_A,d_B,N,tol,d_flag);
        cudaMemcpy(&h_keep_going,d_flag,sizeof(int),cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(ev_e);
    cudaEventSynchronize(ev_e);
    float base1_ms=0.0f;
    cudaEventElapsedTime(&base1_ms,ev_s,ev_e);

    *out_steps=N_conv;
    cudaEventDestroy(ev_s);
    cudaEventDestroy(ev_e);
    return base1_ms;
}
