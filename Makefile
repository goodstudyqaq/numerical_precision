GPU_CC = nvcc
GPU_FLAGS = -O3 -arch=sm_89
CPU_CC = g++
CPU_FLAGS = -O3

cpu_sum: sum1.cpp
	$(CPU_CC) $(CPU_FLAGS) -o cpu_sum sum1.cpp
gpu_sum: sum2.cu
	$(GPU_CC) $(GPU_FLAGS) -o gpu_sum sum2.cu
gen_data: gen_data.cpp
	$(CPU_CC) $(CPU_FLAGS) -o gen_data gen_data.cpp
clean:
	rm -f cpu_sum gpu_sum gen_data data.in