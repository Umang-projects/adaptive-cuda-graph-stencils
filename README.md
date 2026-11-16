# Zero-Sync CUDA Conditional `WHILE` Graphs & Hardware-Adaptive $K^*$ Autotuning for Iterative Stencil Solvers

An experimental systems study extending Ekelund et al. (IEEE PDP 2025, `arXiv:2501.09398`) for convergence-checked iterative stencil solvers (**5-Point Jacobi** and **9-Point Moore**) on **NVIDIA L40S (Ada Lovelace, 142 SMs)** and **NVIDIA Tesla T4 (Turing, 40 SMs)** using CUDA 12.4+ / 12.6.

---

## 1. Problem & Core Contributions

In iterative PDE/stencil solvers with a convergence tolerance check, static CUDA Graph batching (Ekelund et al.) launches a graph of $B$ stencil steps, copies/checks the convergence flag on the **host CPU** (`cudaStreamSynchronize`), and decides whether to replay the graph. This creates a hard trade-off:
- Small batch size $B$ $\rightarrow$ low step overshoot, but high CPU-GPU synchronization overhead.
- Large batch size $B$ $\rightarrow$ fewer CPU syncs, but wasted overshoot steps after convergence and **GPU Instruction-Cache (I-Cache) / descriptor thrashing** when too many kernels ($K$) are unrolled into a single graph.

This repository investigates and benchmarks two architectural improvements:

```text
1. EKELUND STATIC GRAPH (CPU-Driven):
   [Launch B-Step Graph] ──► [D2H Sync to CPU] ──► (if not converged) ──► [CPU Re-launches Graph]

2. OUR CONDITIONAL `WHILE` GRAPH (0 CPU Sync):
   CPU launches 1 Graph ──► [GPU WHILE Node] ──► [Run B Steps] ──► [Device-Side Check] ──(loop on GPU!)──┐
                                  ▲                                                                      │
                                  └──────────────────────────────────────────────────────────────────────┘

3. HARDWARE-ADAPTIVE K* I-CACHE AUTOTUNER:
   Instead of blindly unrolling 128 kernels into 1 graph (which thrash T4's smaller frontend/cache),
   sweep K = {2, 4, 8, 16, 32, 64, 128} at startup to pick K* x Reps matching the target GPU.
```

---

## 2. Experimental Results: NVIDIA L40S vs. Tesla T4

### A. Batch Size ($B$) Sweep (`256 x 256` Grid, True Convergence = `1210` Steps)

| Batch Size ($B$) | Steps Executed | L40S: Ekelund Static (CPU Sync) | L40S: `WHILE` Graph (0 CPU Sync) | L40S Speedup | T4: Ekelund Static (CPU Sync) | T4: `WHILE` Graph (0 CPU Sync) | T4 Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2`** | `1210` (`+0`) | `8.051 ms` | `4.190 ms` | **`+48.0%`** | `15.282 ms` | `11.419 ms` | **`+25.3%`** |
| **`4`** | `1212` (`+2`) | `6.549 ms` | **`2.865 ms`** | **`+56.3%`** | `7.541 ms` | **`5.535 ms`** | **`+26.6%`** |
| **`8`** | `1216` (`+6`) | `3.337 ms` | `2.188 ms` | **`+34.4%`** | `5.015 ms` | `3.714 ms` | **`+25.9%`** |
| **`16`** | `1216` (`+6`) | `2.784 ms` | `1.859 ms` | **`+33.2%`** | `3.748 ms` | `3.214 ms` | **`+14.2%`** |
| **`32`** | `1216` (`+6`) | `2.079 ms` | `1.714 ms` | **`+17.5%`** | `3.069 ms` | `2.971 ms` | `+3.2%` |
| **`64`** | `1216` (`+6`) | `1.818 ms` | `1.631 ms` | `+10.3%` | `2.832 ms` | `2.922 ms` | `-3.2%` |
| **`128`** | `1280` (`+70`) | `1.777 ms` | `1.678 ms` | `+5.6%` | `2.968 ms` | `3.168 ms` | `-6.8%` |
| **`256`** | `1280` (`+70`) | `1.708 ms` | `1.660 ms` | `+2.8%` | `2.936 ms` | `3.158 ms` | `-7.6%` |

*(CPU Non-Graph Baseline: `9.368 ms` on L40S, `16.739 ms` on Tesla T4).*

#### **Key Architectural Observation (When `WHILE` Graphs Win vs. Lose):**
- At small batch sizes ($B \in [2, 16]$), eliminating host synchronization (`cudaStreamSynchronize`) yields **`+33.2%` to `+56.3%`** speedup on L40S and **`+14.2%` to `+26.6%`** on T4.
- **Honest Trade-Off at Large $B$ on T4:** Once $B \ge 64$ on Tesla T4 (`sm_75`), CPU syncs drop below 20 total round-trips; at that point, device-side conditional branch evaluation inside the `WHILE` graph adds a slight overhead (`-3.2%` to `-7.6%`) compared to host-replayed static graphs.

---

### B. Pure I-Cache / Graph Descriptor Sweep (`128 x 128` Grid, Total = `256` Steps)

| Kernels / Graph ($K$) | Replays | L40S Build $T_C$ | L40S Exec $T_E$ | T4 Build $T_C$ | T4 Exec $T_E$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `2` | `128` | `0.023 ms` | `0.3441 ms` | `0.046 ms` | `0.3882 ms` |
| `4` | `64` | `0.015 ms` | `0.3067 ms` | `0.032 ms` | `0.3517 ms` |
| `8` | `32` | `0.070 ms` | `0.2732 ms` | `0.047 ms` | **`0.3377 ms`** |
| `16` | `16` | `0.084 ms` | `0.2611 ms` | `0.060 ms` | `0.3458 ms` |
| `32` | `8` | `0.075 ms` | `0.2614 ms` | `0.093 ms` | `0.3409 ms` |
| `64` | `4` | `0.131 ms` | `0.2583 ms` | `0.173 ms` | `0.3454 ms` |
| `128` | `2` | `0.243 ms` | `0.2570 ms` | `0.340 ms` | `0.3795 ms` |
| `256` | `1` | `0.536 ms` | **`0.2564 ms`** | `0.733 ms` | `0.3921 ms` |

- **Hardware Contrast:** On **Tesla T4**, execution time hits an optimum at $K \in [8, 32]$ (`0.338–0.341 ms`) and **degrades by ~16%** as the graph grows to `256` nodes (`0.392 ms`), alongside graph build time $T_C$ jumping to `0.733 ms`. On **L40S**, larger Ada Lovelace caches absorb `256` kernel descriptors smoothly (`0.2564 ms`).

---

### C. Multi-Grid Evaluation: 5-Point Jacobi & 9-Point Moore (`100` Warmups, `10`-Run Avg)

#### **1. NVIDIA Tesla T4 (`sm_75`) — Both `WHILE` & $K^*$ Autotuner Show Gains**

| Stencil | Grid Size | NoGraph Base | Ekelund (`B=16`) | `WHILE` (`B=16`) | `WHILE` Gain | Picked $K^*$ | Ekelund (`128x1`) | Autotune ($K^* \times$ Reps) | I-Cache Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5-Pt Jacobi** | `32x32` | `4.078 ms` | `1.027 ms` | `0.784 ms` | **`+23.7%`** | `32x4` | `0.718 ms` | `0.657 ms` | **`+8.4%`** |
| **5-Pt Jacobi** | `64x64` | `7.894 ms` | `1.973 ms` | `1.561 ms` | **`+20.9%`** | `32x4` | `1.262 ms` | `1.146 ms` | **`+9.2%`** |
| **5-Pt Jacobi** | `128x128` | `10.496 ms` | `2.654 ms` | `2.268 ms` | **`+14.5%`** | `16x8` | `2.055 ms` | `1.864 ms` | **`+9.3%`** |
| **5-Pt Jacobi** | `256x256` | `10.663 ms` | `3.757 ms` | `3.242 ms` | **`+13.7%`** | `32x4` | `2.963 ms` | `2.761 ms` | **`+6.8%`** |
| **5-Pt Jacobi** | `512x512` | `19.721 ms` | `7.750 ms` | `7.403 ms` | `+4.5%` | `16x8` | `7.113 ms` | `6.933 ms` | `+2.5%` |
| **9-Pt Moore** | `32x32` | `3.408 ms` | `0.845 ms` | `0.680 ms` | **`+19.6%`** | `32x4` | `0.551 ms` | `0.496 ms` | **`+10.0%`** |
| **9-Pt Moore** | `64x64` | `7.010 ms` | `1.765 ms` | `1.382 ms` | **`+21.7%`** | `16x8` | `1.277 ms` | `1.182 ms` | **`+7.4%`** |
| **9-Pt Moore** | `128x128` | `10.191 ms` | `2.593 ms` | `2.407 ms` | `+7.2%` | `32x4` | `2.230 ms` | `2.092 ms` | `+6.2%` |
| **9-Pt Moore** | `256x256` | `10.706 ms` | `4.561 ms` | `4.007 ms` | `+12.1%` | `16x8` | `3.795 ms` | `3.611 ms` | `+4.9%` |
| **9-Pt Moore** | `512x512` | `20.849 ms` | `10.861 ms` | `10.496 ms` | `+3.4%` | `16x8` | `10.341 ms` | `10.123 ms` | `+2.1%` |

#### **2. NVIDIA L40S (`sm_89`) — Strong `WHILE` Gains; Autotuner Correctly Selects `128x1`**

| Stencil | Grid Size | NoGraph Base | Ekelund (`B=16`) | `WHILE` (`B=16`) | `WHILE` Gain | Picked $K^*$ | Ekelund (`128x1`) | Autotune ($K^* \times$ Reps) | I-Cache Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5-Pt Jacobi** | `32x32` | `3.419 ms` | `0.874 ms` | `0.552 ms` | **`+36.9%`** | `128x1` | `0.519 ms` | `0.516 ms` | `+0.5%` |
| **5-Pt Jacobi** | `64x64` | `6.619 ms` | `1.683 ms` | `1.078 ms` | **`+36.0%`** | `128x1` | `0.961 ms` | `0.959 ms` | `+0.1%` |
| **5-Pt Jacobi** | `128x128` | `8.886 ms` | `2.247 ms` | `1.508 ms` | **`+32.9%`** | `128x1` | `1.451 ms` | `1.451 ms` | `-0.0%` |
| **5-Pt Jacobi** | `256x256` | `9.002 ms` | `2.830 ms` | `1.865 ms` | **`+34.1%`** | `128x1` | `1.780 ms` | `1.787 ms` | `-0.4%` |
| **5-Pt Jacobi** | `512x512` | `9.073 ms` | `3.331 ms` | `2.484 ms` | **`+25.4%`** | `128x1` | `2.361 ms` | `2.365 ms` | `-0.2%` |
| **9-Pt Moore** | `32x32` | `3.230 ms` | `0.776 ms` | `0.466 ms` | **`+39.9%`** | `128x1` | `0.402 ms` | `0.402 ms` | `+0.0%` |
| **9-Pt Moore** | `64x64` | `6.697 ms` | `1.605 ms` | `0.980 ms` | **`+38.9%`** | `128x1` | `0.977 ms` | `0.975 ms` | `+0.2%` |
| **9-Pt Moore** | `128x128` | `9.762 ms` | `2.347 ms` | `1.504 ms` | **`+35.9%`** | `128x1` | `1.471 ms` | `1.472 ms` | `-0.1%` |
| **9-Pt Moore** | `256x256` | `10.237 ms` | `2.466 ms` | `1.891 ms` | **`+23.3%`** | `128x1` | `1.798 ms` | `1.803 ms` | `-0.3%` |
| **9-Pt Moore** | `512x512` | `10.322 ms` | `3.129 ms` | `2.528 ms` | **`+19.2%`** | `128x1` | `2.457 ms` | `2.461 ms` | `-0.1%` |

*(Note how the hardware-adaptive autotuner dynamically picks `16x8` / `32x4` on T4 for up to `+10.0%` gain while picking `128x1` on L40S where large graph unrolling does not penalize execution).*

---

## 3. Repository Layout & Reproducing Benchmarks

```text
.
├── test_paper_final.cu    # Unified benchmark suite generating Tables 1–4
├── modal_run.py           # Modal runner (CUDA 12.6.0, T4 & L40S)
└── b_star_predictor/      # Modular C++ & CUDA analytical B* predictor
    ├── solver.hpp
    ├── solver.cu
    └── main.cpp
```

### Run Locally (CUDA 12.4+ Required for `cudaGraphCondTypeWhile`)
```bash
nvcc -O3 test_paper_final.cu -o test_paper_final
./test_paper_final
```

### Run via Modal
```bash
modal run modal_run.py
```

---

## References
- Ekelund et al., *"Static CUDA Graphs for Iterative Stencil Computations"*, IEEE PDP 2025 (`arXiv:2501.09398`).