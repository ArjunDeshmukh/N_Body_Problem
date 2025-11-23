#ifndef DEVICE_BUFFER_GPU_H_
#define DEVICE_BUFFER_GPU_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>   // Optional but useful on Windows

// Add this near the top or in a header
const int MAX_BODIES = 20000; // Hard limit to prevent re-allocation complexity

struct DeviceBuffers {
    double *d_px, *d_py, *d_mass, *d_ax, *d_ay;
    
    void allocate() {
        size_t ds = MAX_BODIES * sizeof(double);
        cudaMalloc(&d_px, ds);
        cudaMalloc(&d_py, ds);
        cudaMalloc(&d_mass, ds);
        cudaMalloc(&d_ax, ds);
        cudaMalloc(&d_ay, ds);
    }

    void free() {
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_mass);
        cudaFree(d_ax); cudaFree(d_ay);
    }
};

#endif // DEVICE_BUFFER_GPU_H_