
#ifndef _pmm_utils_h
#define _pmm_utils_h

#include "PerformanceMeasure.cuh"
//#include "Utility.h"
#include "cuda.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <cassert>

using namespace std;
//#include "src/gpu_hash_table.cuh"


extern "C"{

//#define DEBUG
#ifdef DEBUG
#define debug(a...) printf(a)
#else
#define debug(a...)
#endif

#define EMPTY 0
#define DONE  2
#define MOCK  9

enum request_type {
    request_empty  = EMPTY,
    request_done   = DONE,
    reqeust_mock   = MOCK
};
cudaError_t GRError(cudaError_t error, const char *message,
                    const char *filename, int line, bool print) {
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename,
            line, gpu, message, error, cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

#define GUARD_CU(cuda_call)                                                   \
  {                                                                           \
    if (cuda_call != (enum cudaError) CUDA_SUCCESS){  \
        printf("--- ERROR(%d:%s) --- %s:%d\n", cuda_call, cudaGetErrorString(cuda_call), __FILE__, __LINE__);\
    } \
  }\

struct RequestType{

    volatile int* requests_number; 
    volatile int* request_counter;
    volatile int* request_signal; 
    volatile int* request_id; 
    volatile int* request_mem_size;
    volatile int* lock;
    volatile int** d_memory{nullptr};
    volatile int** request_dest;
    int size;

    void init(size_t Size);
    void memset();
    void free();

};

void RequestType::init(size_t Size){

    size = Size;

    GUARD_CU(cudaMallocManaged(&requests_number,         sizeof(volatile int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaMallocManaged(&request_counter,         sizeof(volatile int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaMallocManaged(&request_signal,   size * sizeof(volatile int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaMallocManaged(&request_id,       size * sizeof(volatile int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaMallocManaged(&request_mem_size, size * sizeof(volatile int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaMallocManaged(&lock,             size * sizeof(volatile int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaMallocManaged(&d_memory,         size * sizeof(volatile int*)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaMallocManaged(&request_dest,     size * sizeof(volatile int*)));
    GUARD_CU(cudaPeekAtLastError());

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    /*
    uint32_t num_keys = 1<<20;
    float expected_chain = 0.6f;
    uint32_t num_elements_per_uint = 15;
    uint32_t expected_elements_per_bucket = expected_chain * num_elements_per_uint;
    uint32_t num_buckets = (num_keys + expected_elements_per_bucket - 1)/expected_elements_per_bucket;
    const int64_t seed = 1;
    uint32_t DEVICE_ID = 0;
    hash_table = 
        new gpu_hash_table<uint32_t, uint32_t, SlabHashTypeT::ConcurrentMap>(num_keys, num_buckets, DEVICE_ID, seed);
    */

}

void RequestType::free(){

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)requests_number));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)request_counter));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)request_signal));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)request_id));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)request_mem_size));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)lock));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)d_memory));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)request_dest));
    GUARD_CU(cudaPeekAtLastError());

    //delete(hash_table);

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void RequestType::memset(){

    *requests_number = size;

    *request_counter = 0;

    GUARD_CU(cudaMemset((void*)request_signal, 0,   size * sizeof(volatile int)));
    GUARD_CU(cudaMemset((void*)request_id, -1,      size * sizeof(volatile int)));
    GUARD_CU(cudaMemset((void*)request_mem_size, 0, size * sizeof(volatile int)));
    GUARD_CU(cudaMemset((void*)lock, 0,             size * sizeof(volatile int)));
    GUARD_CU(cudaMemset((int**)d_memory, 0,         size * sizeof(volatile int*)));
    GUARD_CU(cudaMemset((int**)request_dest, 0,     size * sizeof(volatile int*)));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}
/*
__global__
void copy(int** d_memory0, int* d_memory, int size){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    d_memory[thid] = d_memory0[thid][0];
}

//mem test
void mem_test(int** d_memory0, int requests_num, int blocks, int threads){
    //create array
    int* d_memory{nullptr};
    GUARD_CU(cudaMalloc(&d_memory, sizeof(int) * requests_num));
    GUARD_CU(cudaPeekAtLastError());
    copy<<<blocks, threads>>>(d_memory0, d_memory, requests_num);
    GUARD_CU(cudaStreamSynchronize(0));
    GUARD_CU(cudaPeekAtLastError());
    int* h_memory = (int*)malloc(requests_num* sizeof(int));
    GUARD_CU(cudaMemcpy(h_memory, d_memory, sizeof(int)*requests_num, cudaMemcpyDeviceToHost));
    GUARD_CU(cudaStreamSynchronize(0));
    GUARD_CU(cudaPeekAtLastError());
}*/

__device__
inline int thid(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}

//__device__ void acquire_semaphore(volatile int* lock, int i){
//    while (atomicCAS((int*)&lock[i], 0, 1) != 0){
//        //printf("acq semaphore: thread %d\n", threadIdx.x);
//    }
//    __threadfence();
//}

__device__ 
void acquire_semaphore(int* lock){
    while (atomicCAS(lock, 0, 1) != 0){
        //printf("acq semaphore: thread %d\n", threadIdx.x);
    }
    __threadfence();
}

__device__ 
void release_semaphore(int* lock){
    __threadfence();
    *lock = 0;
}
//__device__ void release_semaphore(volatile int* lock, int i){
//    __threadfence();
//    lock[i] = 0;
//}
}
#endif
