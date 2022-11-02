
#ifndef _pmm_utils_h
#define _pmm_utils_h

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "PerformanceMeasure.cuh"
#include "Utility.h"
#include "cuda.h"
#include "cuda_profiler_api.h"
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

#ifdef HALLOC__
#include "Instance.cuh"
#endif


#ifdef OUROBOROS__
    //Ouroboros initialization
    #define MemoryManagerType OuroPQ
#endif
#ifdef HALLOC__
    //Halloc initialization
    #define MemoryManagerType MemoryManagerHallokldfsdfjkkj
#endif

#define EMPTY       0
#define DONE        2
#define MALLOC      3
#define FREE        5
#define GC          7

#define MPS           0
#define MPS_mono      1
#define simple_mono   2
#define one_per_warp  3
#define async_request 4

enum request_type {
    request_empty       = EMPTY,
    request_done        = DONE,
    request_malloc      = MALLOC, 
    request_free        = FREE,
    request_gc          = GC
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

struct request{
    int id;
    int signal;
    int mem_size;
    int lock;
    volatile int* dest;

    request(): id(-1), signal(0), mem_size(0), lock(0), dest(NULL){}
    request(int Id, int Signal, int Mem_size, int Lock, volatile int* Dest): 
                id(Id), signal(Signal), mem_size(Mem_size), lock(Lock), dest(Dest){}

    volatile int* get(){
        return dest;
    }

};

struct Runtime{

    volatile int* requests_number; 
    volatile int* request_counter;
    volatile int** d_memory{nullptr};
    request* data;
    int size;

    Runtime(): requests_number(NULL), request_counter(NULL), d_memory(NULL){}
    void init(size_t Size);
    void memset();
    void free();
};

/*
    Size - the number of threads assigned to the application
           One reqeust per thread at a time
 */
void Runtime::init(size_t Size){
    size = Size;
    GUARD_CU(cudaMallocManaged(&requests_number, sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&request_counter, sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&data, size * sizeof(request)));
}

void Runtime::free(){
    GUARD_CU(cudaFree((void*)requests_number));
    GUARD_CU(cudaFree((void*)request_counter));
    GUARD_CU(cudaFree((void*)data));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void Runtime::memset(){

    *requests_number = size;
    *request_counter = 0;
    for (int i=0; i<size; ++i){
        new(data + i) request(-1, 0, 0, 0, NULL);
    }
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    
}

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

/*
void RequestType::init(size_t Size){
    size = Size;
    GUARD_CU(cudaMallocManaged(&requests_number, sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&request_counter, sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&request_signal, size * sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&request_id, size * sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&request_mem_size, size * sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&lock, size * sizeof(volatile int)));
    GUARD_CU(cudaMallocManaged(&d_memory, size * sizeof(volatile int*)));
    GUARD_CU(cudaMallocManaged(&request_dest, size * sizeof(volatile int*)));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}
*/

void RequestType::init(size_t Size){
    size = Size;
    GUARD_CU(cudaMalloc((void**)&requests_number, sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&request_counter, sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&request_signal, size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&request_id, size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&request_mem_size, size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&lock, size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&d_memory, size * sizeof(volatile int*)));
    GUARD_CU(cudaMalloc((void**)&request_dest, size * sizeof(volatile int*)));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void RequestType::free(){

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaFree((void*)requests_number));
    GUARD_CU(cudaFree((void*)request_counter));
    GUARD_CU(cudaFree((void*)request_signal));
    GUARD_CU(cudaFree((void*)request_id));
    GUARD_CU(cudaFree((void*)request_mem_size));
    GUARD_CU(cudaFree((void*)lock));
    GUARD_CU(cudaFree((void*)d_memory));
    GUARD_CU(cudaFree((void*)request_dest));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void RequestType::memset(){

    //*requests_number = size;
    //*request_counter = 0;
    int zero = 0;
    int* tab_zero = (int*)malloc(size * sizeof(int));
    for (int i=0; i<size; ++i) tab_zero[i] = 0;
    GUARD_CU(cudaMemcpy((void*)requests_number, &size, sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)request_counter, &zero, sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)request_signal,     tab_zero, size * sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)request_mem_size,   tab_zero, size * sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)lock,               tab_zero, size * sizeof(int), cudaMemcpyHostToDevice));
    for (int i=0; i<size; ++i) tab_zero[i] = -1;
    GUARD_CU(cudaMemcpy((void*)request_id,         tab_zero, size * sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    
}

/*
void RequestType::memset(){
    requests_number = size;
    request_counter = 0;
    GUARD_CU(cudaMemset((void*)request_signal, 0, size * sizeof(volatile int)));
    GUARD_CU(cudaMemset((void*)request_id, -1, size * sizeof(volatile int)));
    GUARD_CU(cudaMemset((void*)request_mem_size, 0, size * sizeof(volatile int)));
    GUARD_CU(cudaMemset((void*)lock, 0, size * sizeof(volatile int)));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}
*/

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
}

__device__ void acquire_semaphore(volatile int* lock, int i){
    while (atomicCAS((int*)&lock[i], 0, 1) != 0){
        //printf("acq semaphore: thread %d\n", threadIdx.x);
    }
    __threadfence();
}

__device__ void release_semaphore(volatile int* lock, int i){
    __threadfence();
    lock[i] = 0;
}

__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned sm_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

//test
__global__
void test1(volatile int** d_memory, int size){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid < size){
        assert(d_memory[thid]);
        d_memory[thid][0] *= 100;
    }
}

__global__
void test2(volatile int** d_memory, int size){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid < size){
        assert(d_memory[thid] == NULL);
    }
}
}
#endif
