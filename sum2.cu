#include <bits/stdc++.h>
using namespace std;
#define real double

__global__ void sum_atomic_kernel(const real* data, real* sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(sum, data[idx]);
    }
}

real gpu_sum_atomic(const vector<real>& h_data) {
    int n = h_data.size();
    real* d_data;
    real* d_sum;
    cudaMalloc(&d_data, n * sizeof(real));
    cudaMalloc(&d_sum, sizeof(real));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(real));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sum_atomic_kernel<<<numBlocks, blockSize>>>(d_data, d_sum, n);
    real h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_sum);
    return h_sum;
}


__global__ void tree_reduce_kernel(const real* in, real* out, int n) {
    extern __shared__ real sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;


    real x = 0.0;
    if (i < n) x += in[i];
    if (i + blockDim.x < n) x += in[i + blockDim.x];

    sdata[tid] = x;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}



real gpu_sum_tree_reduce(const vector<real>& h_data) {
    int n = h_data.size();
    if (n == 0) return 0.0;

    const int threads = 256;

    real *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(real));
    cudaMemcpy(d_in, h_data.data(), n * sizeof(real), cudaMemcpyHostToDevice);

    int cur_n = n;
    real* cur_in = d_in;

    // One block has 256 threads, so it can reduce 256 * 2 = 512 elements in one step

    int max_blocks = (cur_n + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&d_out, max_blocks * sizeof(real));

    while (cur_n > 1) { 
        int blocks = (cur_n + threads * 2 - 1) / (threads * 2);
        size_t shared_mem_size = threads * sizeof(real);

        tree_reduce_kernel<<<blocks, threads, shared_mem_size>>>(cur_in, d_out, cur_n);
        cudaDeviceSynchronize();

        cur_n = blocks;
        cur_in = d_out;
    }

    real result;
    cudaMemcpy(&result, cur_in, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    return result;
}



int main() {
    freopen("data.in", "r", stdin);
    int n;
    cin >> n;
    vector<real> data(n);
    for (int i = 0; i < n; i++) {
        cin >> data[i];
    }

    for (int t = 0; t < 10; t++) {
        real sum = gpu_sum_atomic(data);
        cout << "Test " << t << ": Atomic sum = " << fixed << setprecision(10) << sum << endl;
    }

    for (int t = 0; t < 10; t++) {
        real sum = gpu_sum_tree_reduce(data);
        cout << "Test " << t << ": Tree reduce sum = " << fixed << setprecision(10) << sum << endl;
    }


/* 
thread size = 256 每个 block 有 256 个线程
block size = prop.multiprocessor_count * 8 每个 grid 有 prop.multiprocessor_count * 8 个 block

blockDim.x = 256 
gridDim.x = prop.multiprocessor_count * 8

*/
}