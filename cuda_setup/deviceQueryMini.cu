#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("CUDA Devices: %d\n", count);
    for (int i = 0; i < count; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("Device %d: %s\n", i, p.name);
    }
    return 0;
}
