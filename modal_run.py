import subprocess
import modal

app=modal.App("while-graph-brute-step1")

image=(
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04",add_python="3.11")
    .pip_install("torch","transformers","accelerate","ninja")
    .add_local_dir(".",remote_path="/root/project")
)

# @app.function(image=image,gpu="T4",timeout=300)
# def run():
#     print("[1/3] Compiling solver.cu with nvcc -> solver.o...")
#     subprocess.run(["nvcc","-O3","-c","solver.cu","-o","solver.o"],cwd="/root/project",check=True)
# 
#     print("[2/3] Compiling main.cpp with g++ -> main.o...")
#     subprocess.run(["g++","-O3","-c","main.cpp","-I/usr/local/cuda/include","-o","main.o"],cwd="/root/project",check=True)
# 
#     print("[3/3] Linking main.o + solver.o -> app...")
#     subprocess.run(["g++","main.o","solver.o","-L/usr/local/cuda/lib64","-lcudart","-o","app"],cwd="/root/project",check=True)
# 
#     subprocess.run(["./app"],cwd="/root/project",check=True)
# 
# @app.local_entrypoint()
# def main():
#     run.remote()


@app.function(image=image,gpu="T4",timeout=300)
def run():
    # subprocess.run(["nvcc","-O3","toy_runner.cu","-o","toy_runner"],cwd="/root/project",check=True)
    # subprocess.run(["./toy_runner"],cwd="/root/project",check=True)
    # subprocess.run(["nvcc","-O3","test_cachepipeline.cu","-o","test_cache_pipeline"],cwd="/root/project",check=True)
    # subprocess.run(["./test_cache_pipeline"],cwd="/root/project",check=True)
    # modal_run.py ke andar run() function mein:
    # subprocess.run(["nvcc","-O3","test_ekelund_vs_ours.cu","-o","test_ekelund"],cwd="/root/project",check=True)
    # subprocess.run(["./test_ekelund"],cwd="/root/project",check=True)
    # modal_run.py ke andar run() function mein:
    # subprocess.run(["nvcc","-O3","test_multi_grid.cu","-o","test_multi_grid"],cwd="/root/project",check=True)
    # subprocess.run(["./test_multi_grid"],cwd="/root/project",check=True)
    # modal_run.py ke andar run() function mein:
    # subprocess.run(["nvcc","-O3","test_icache_sweep.cu","-o","test_icache_sweep"],cwd="/root/project",check=True)
    # subprocess.run(["./test_icache_sweep"],cwd="/root/project",check=True)
    subprocess.run(["nvcc","-O3","test_paper_final.cu","-o","test_paper_final"],cwd="/root/project",check=True)
    subprocess.run(["./test_paper_final"],cwd="/root/project",check=True)

@app.local_entrypoint()
def main():
    run.remote()
