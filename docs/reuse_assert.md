## Reordered GEMM

### Runtime (ms)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         |    |    |    |    |       |
| 256         |    |    |    |    |       |
| 512         |    |    |    |    |       |
| 1024        |    |    |    |    |       |

### Performance (GFLOPS)

| Matrix Size | O0 | O1 | O2 | O3 | Ofast |
|-------------|----|----|----|----|-------|
| 128         |    |    |    |    |       |
| 256         |    |    |    |    |       |
| 512         |    |    |    |    |       |
| 1024        |    |    |    |    |       |




如何启动？（样例）

常规测试
cd build
cmake ..
cmake --build .


cmake -S . -B build-O0 -DOPT_LEVEL=O0
cmake --build build-O0

./build-O0/gemm_benchmark


cmake -S . -B build-O1 -DOPT_LEVEL=O1
cmake --build build-O1

./build-O1/gemm_benchmark



cmake -S . -B build-O2 -DOPT_LEVEL=O2
cmake --build build-O2

./build-O2/gemm_benchmark


cmake -S . -B build-O3 -DOPT_LEVEL=O3
cmake --build build-O3

./build-O3/gemm_benchmark



cmake -S . -B build-Ofast -DOPT_LEVEL=Ofast
cmake --build build-Ofast

./build-Ofast/gemm_benchmark





测试 Operators的样例
python tests/python/generate_softmax_input.py
cmake --build build-O0

./build-O0/ops_test
python tests/python/test_softmax_reference.py




创建新的服务器实例的时候用。
# 1. 检查 GPU / Driver
nvidia-smi

# 2. 先看看 nvcc 是否已经在 PATH
which nvcc
nvcc --version

# 3. 如果 nvcc 找不到，检查 CUDA 安装目录
ls /usr/local
ls /usr/local/cuda*
find /usr/local -name nvcc 2>/dev/null

# 4. 临时加入 CUDA PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 5. 永久写入 ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 6. 立即生效
source ~/.bashrc

# 7. 再次确认 CUDA compiler
which nvcc
nvcc --version

# 8. 检查 Nsight 工具
ncu --version
nsys --version

# 9. 检查 C++ / CMake / Git 环境
g++ --version
cmake --version
git --version


# git安装工具
apt update
apt install -y build-essential cmake git



启动单独测试项
cmake --build build --target cuda_vector_add_test
./build/cuda_vector_add_test


-S . 代表源目录位置在.  -B 生成出来的构建文件放到哪里。
cmake -S . -B build   




nsys profile \
    --trace=cuda \
    --sample=none \
    --cpuctxsw=none \
    -o softmax_multi_8192 \
    ./build/cuda_attention_softmax_profile 8192 multi 4
