#include "solver.hpp"

struct TestCase{
    const char* name;
    int N;
    float tol;
};

int main(){
    int *d_flag,*d_steps;
    cudaMalloc(&d_flag,sizeof(int));
    cudaMalloc(&d_steps,sizeof(int));

    // 3 Bilkul Alag-Alag Cases (Small B*, Medium B*, Large B* Regimes!)
    TestCase cases[3]={
        {"CASE 1: Heavy 1024x1024 Grid (Fast Settle, tol=0.25)", 1024, 0.25f},
        {"CASE 2: Medium 256x256 Grid  (Medium Settle, tol=0.02)", 256, 0.02f},
        {"CASE 3: Tight 256x256 Grid   (Deep Settle, tol=0.003)",  256, 0.003f}
    };

    int B_list[8]={2,4,8,16,32,64,128,256};

    for(int c=0;c<3;c++){
        int N=cases[c].N;
        float tol=cases[c].tol;
        int bytes=N*N*sizeof(float);

        float* h_init=new float[N*N]();
        for(int x=0;x<N;x++){ h_init[x]=100.0f; }

        float *d_A,*d_B;
        cudaMalloc(&d_A,bytes);
        cudaMalloc(&d_B,bytes);

        // 1. Micro-Probe
        float t_step_us=0.0f,t_check_us=0.0f;
        probe_timings(d_A,d_B,h_init,N,tol,d_flag,d_steps,&t_step_us,&t_check_us);

        // 2. Baseline 1 (CPU Loop)
        int N_conv=0;
        float base1_ms=run_cpu_loop(d_A,d_B,h_init,N,tol,d_flag,&N_conv);

        // 3. Formula Prediction: B* = sqrt(2 * N * t_check / t_step)
        float raw_B=sqrtf((2.0f*N_conv*t_check_us)/t_step_us);
        int B_star=((int)roundf(raw_B/2.0f))*2;
        if(B_star<2) B_star=2;

        printf("\n========================================================================\n");
        printf("  %s\n",cases[c].name);
        printf("========================================================================\n");
        printf("  • True Convergence Steps (N)   : %d steps\n",N_conv);
        printf("  • Single Step Time (t_step)    : %.2f us\n",t_step_us);
        printf("  • GPU Check Time (t_check)     : %.2f us\n",t_check_us);
        printf("  • Baseline 1 (CPU Loop) Time   : %.3f ms\n",base1_ms);
        printf("  • FORMULA PREDICTED OPTIMAL B* : B* = %d steps (raw = %.1f)\n",B_star,raw_B);
        printf("------------------------------------------------------------------------\n");
        printf("   B   | Steps Executed | Overshoot | WHILE Graph Time | Speedup vs CPU\n");
        printf("-------+----------------+-----------+------------------+----------------\n");

        int best_B=0;
        float best_ms=1e9f;

        for(int i=0;i<8;i++){
            int B=B_list[i];
            int steps_ran=0;
            run_while_graph(B,d_A,d_B,h_init,N,tol,d_flag,d_steps,&steps_ran);
            float t_ms=run_while_graph(B,d_A,d_B,h_init,N,tol,d_flag,d_steps,&steps_ran);
            int overshoot=steps_ran-N_conv;

            if(t_ms<best_ms){ best_ms=t_ms; best_B=B; }

            printf("  %3d  |   %5d steps  |  +%3d stp |     %7.3f ms   |     %5.2fx\n",
                   B,steps_ran,overshoot,t_ms,base1_ms/t_ms);
        }
        printf("------------------------------------------------------------------------\n");
        printf("  🏆 ACTUAL HARDWARE WINNER: B = %d (%.3f ms) | FORMULA PREDICTED: B* = %d\n",
               best_B,best_ms,B_star);
        printf("========================================================================\n");

        delete[] h_init;
        cudaFree(d_A);
        cudaFree(d_B);
    }

    cudaFree(d_flag);
    cudaFree(d_steps);
    return 0;
}
