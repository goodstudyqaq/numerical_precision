# Numerical Precision Playground

This mini-project demonstrates how floating-point precision affects large-scale summations on the CPU and GPU. It ships three CPU implementations (sequential, thread-parallel, and fixed-point) alongside two CUDA kernels (atomic accumulation and shared-memory tree reduction). All programs consume the same `data.in` file so you can compare accuracy and throughput across approaches.

## Project Layout

| Path | Description |
| --- | --- |
| `sum1.cpp` → `cpu_sum` | CPU experiments using `double` precision (sequential, 1K-threaded, and fixed-point accumulation). |
| `sum2.cu` → `gpu_sum` | CUDA experiments using `float` precision (global atomic vs. shared-memory tree reduction). |
| `gen_data.cpp` → `gen_data` | Deterministic generator for `data.in` (10M uniformly distributed values in ±1e5 by default). |
| `Makefile` | Convenience targets for building the host, device, and data-generation binaries. |

## Prerequisites

- Linux environment with `g++` (C++17) and `nvcc` targeting an SM 8.9 GPU by default.
- CUDA-capable GPU with enough memory to hold ~10 million `float`s (≈40 MB) and shared memory per block ≥ 48 KB.
- Optional: Modify `GPU_FLAGS` in the `Makefile` if your GPU uses a different compute capability (e.g., `-arch=sm_86`).

## Building

```bash
make all        # Builds gen_data, cpu_sum, gpu_sum
make gen_data   # Only the data generator
make cpu_sum    # Only the CPU executable
make gpu_sum    # Only the GPU executable
make clean      # Remove binaries and data.in
```

The `Makefile` applies `-O3` optimizations to both host and device code. Adjust `CPU_FLAGS` / `GPU_FLAGS` to experiment with alternative compiler settings.

## Generating Input Data

```bash
./gen_data
```

This writes a fresh `data.in` with the default configuration:

- `n = 10,000,000` samples.
- Each sample is drawn uniformly from `[-1e5, 1e5]` and printed with 10 decimal places.

To explore different ranges or sample counts, edit the constants in `gen_data.cpp`, recompile, and regenerate the file.

## Running the Experiments

Both executables read `data.in` from the current working directory. Make sure `gen_data` has been run at least once.

### CPU (`./cpu_sum`)

The CPU binary runs three loops (10 trials each):

1. **Sequential double sum** – single-threaded accumulation.
2. **Thread-parallel sum** – up to 1,000 worker threads, each summing a block, then combining under a mutex.
3. **Fixed-point emulation** – values scaled by 2³² into `long long` before accumulation to reduce rounding error.

### GPU (`./gpu_sum`)

The GPU binary also launches twice (10 trials each):

1. **Atomic-add kernel** – every thread atomically adds into a single global `float`.
2. **Tree-reduction kernel** – block-level shared-memory reduction followed by iterative grid reductions until one value remains.

## Sample Output

```
Test 1: Sequential sum = -149.5321849200
...
Test 1: Parallel(threads = 1000) sum = -149.5321807861
...
Test 1: Fixed-point sum = -149.5321848815
...
Test 0: Atomic sum = -149.5321807861
...
Test 0: Tree reduce sum = -149.5321816206
```

The absolute values will shift with each regenerated dataset, but the relative drift between methods illustrates how different accumulation strategies behave under rounding error.

## Experiment Ideas

- Compare `float` vs. `double` on the CPU by toggling the `real` typedef.
- Adjust `num_threads` in the CPU parallel path to see when contention overtakes throughput.
- Change `blockSize` or shared-memory usage in the CUDA kernels to observe occupancy effects.
- Increase the magnitude/variance of the generated numbers to exacerbate rounding noise.

## Cleaning Up

```bash
make clean
```

This removes all binaries and the generated `data.in`, letting you start from scratch.
