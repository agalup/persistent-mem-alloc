
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

#define EMPTY       0
#define DONE        2
#define MALLOC      3
#define FREE        5
#define GC          7

#define MPS                0
#define MPS_mono           1
#define simple_mono        2
#define one_per_warp       3
#define async_request      4
#define async_one_per_warp 5

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

void allocManaged(volatile int** ptr, size_t size){
    GUARD_CU(cudaMallocManaged(ptr, size));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
}

void allocManaged_(int** ptr, size_t size){
    GUARD_CU(cudaMallocManaged(ptr, size));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
}

struct RequestType{
    volatile size_t* requests_number; 
    volatile int* request_counter;
    volatile int* request_signal; 
    volatile int* request_id; 
    volatile int* request_mem_size;
    volatile int* lock;
    volatile int** d_memory;
    volatile int** request_dest;

    void init(size_t Size);
    void memset(size_t Size);
    void free();
    
    __forceinline__ __device__ size_t number(){ return requests_number[0]; }
    __forceinline__ __device__ int type(int id){ return request_signal[id]; }
};

void RequestType::init(size_t Size){
    GUARD_CU(cudaMalloc((void**)&requests_number, sizeof(volatile size_t)));
    GUARD_CU(cudaMalloc((void**)&request_counter, sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&request_signal, Size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&request_id, Size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&request_mem_size, Size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&lock, Size * sizeof(volatile int)));
    GUARD_CU(cudaMalloc((void**)&d_memory, Size * sizeof(volatile int*)));
    GUARD_CU(cudaMalloc((void**)&request_dest, Size * sizeof(volatile int*)));
    
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void RequestType::memset(size_t Size){
    int zero = 0;
    int* tab_zero = (int*)std::malloc(Size * sizeof(int));
    volatile int** null_tab = (volatile int**)std::malloc(Size * sizeof(volatile int*));
    for (int i=0; i<Size; ++i){
        tab_zero[i] = 0;
        null_tab[i] = NULL;
    }
    GUARD_CU(cudaMemcpy((void*)d_memory, (void*)null_tab, Size * sizeof(volatile int*), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)request_dest, null_tab, Size * sizeof(volatile int*), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)requests_number, &Size, sizeof(size_t), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)request_counter, &zero, sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)request_signal, tab_zero, Size * sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)request_mem_size, tab_zero, Size * sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaMemcpy((void*)lock, tab_zero, Size * sizeof(int), cudaMemcpyHostToDevice));
    for (int i=0; i<Size; ++i) tab_zero[i] = -1;
    GUARD_CU(cudaMemcpy((void*)request_id, tab_zero, Size * sizeof(int), cudaMemcpyHostToDevice));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void RequestType::free(){
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

struct Service{
    volatile int* started;   

    //HOST
    void init(CUdevice device);
    __forceinline__ int is_running(){ return started[0]; }
    //DEVICE
    __forceinline__ __device__ void start(){ *started = 1; }
};

void Service::init(CUdevice device){
    GUARD_CU(cudaMallocManaged(&started, sizeof(uint32_t)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
    *started = 0;
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)started, sizeof(int), device, NULL));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

struct Runtime{
    int app_threads_num;
    volatile int* exit_signal;
    volatile int* exit_counter; 
    Service* mm;
    Service* gc;
    Service* cb;
    RequestType* requests;

    //HOST
    void init(size_t Size, CUdevice device);
    void free();
    __forceinline__ void stop(){ *exit_signal = 1; }

    //DEVICE
    __forceinline__ __device__ size_t size(){ return requests->requests_number[0]; }
    __forceinline__ __device__ volatile int is_working(){ 
        return (! exit_signal[0]); 
        }
    __forceinline__ __device__ volatile int is_available(int thid){ 
        return (requests->request_signal[thid] == request_empty); 
        }
    __forceinline__ __device__ volatile int is_finished(int thid){ 
        return (requests->request_signal[thid] == request_done); 
        }
    __forceinline__ __device__ int type(int thid){
        return (requests->request_signal[thid]);
    }
    __device__ void malloc(volatile int**, size_t);
    __device__ void malloc_async(volatile int**, size_t);
    __device__ void free(volatile int*);
    __device__ void free_async(volatile int**);
    __device__ void request(request_type, volatile int**, int);
    __device__ void request_async(request_type, volatile int**, int);
    __device__ void post_request(request_type, int);
    __device__ void request_processed(request_type, volatile int**);
    __device__ void _request_processing(int, MemoryManagerType*); 
};

void Runtime::init(size_t APP_threads_number, CUdevice device){
    app_threads_num = APP_threads_number;
    allocManaged(&exit_signal, sizeof(int32_t));
    allocManaged(&exit_counter, sizeof(uint32_t));

    GUARD_CU(cudaMallocManaged(&mm, sizeof(Service)));
    GUARD_CU(cudaMallocManaged(&gc, sizeof(Service)));
    GUARD_CU(cudaMallocManaged(&cb, sizeof(Service)));
    GUARD_CU(cudaMallocManaged(&requests, sizeof(RequestType)));

    *exit_signal = 0;
    *exit_counter = 0;

    assert(mm);
    assert(gc);
    assert(cb);
    assert(requests);

    mm->init(device);
    gc->init(device);
    cb->init(device);
    requests->init(APP_threads_number);
    requests->memset(APP_threads_number);

    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_signal, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_counter, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((Service*)mm, sizeof(Service), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((Service*)gc,  sizeof(Service), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((Service*)cb, sizeof(Service), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((RequestType*)requests, sizeof(RequestType), device, NULL));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    printf("request addr %x\n", requests);
}

void Runtime::free(){
    GUARD_CU(cudaFree((void*)exit_signal));
    GUARD_CU(cudaFree((void*)exit_counter));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

__device__
void Runtime::post_request(request_type type, int size_to_alloc){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // SEMAPHORE
    __threadfence();
    acquire_semaphore((int*)(requests->lock), thid);
    if (type == MALLOC){
        requests->request_mem_size[thid] = size_to_alloc;
    }
    // SIGNAL update
    atomicExch((int*)&(requests->request_signal[thid]), type);
    debug("APP %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, thid, blockIdx.x, warp_id(), lane_id(), sm_id());
    release_semaphore((int*)(requests->lock), thid);
    __threadfence();
    // SEMAPHORE
}

__device__
void Runtime::request_processed(request_type type,
                      volatile int** dest){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int req_id = -1;
    // SEMAPHORE
    __threadfence();
    acquire_semaphore((int*)(requests->lock), thid);
    debug("APP %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, thid, blockIdx.x, warp_id(), lane_id(), sm_id());
    switch (type){
        case MALLOC:
            req_id = requests->request_id[thid];
            if (req_id >= 0 && !exit_signal[0]) {
                *dest = requests->request_dest[thid];
                assert(requests->d_memory[req_id] != NULL);
                if (requests->d_memory[req_id][0] != 0)
                    printf("d_memory[%d] = %d\n", req_id, requests->d_memory[req_id][0]);
                //assert(d_memory[req_id][0] == 0);
                assert(*dest != NULL);
                assert(requests->request_dest[thid] == *dest);
            }
            break;
        case FREE:
            //assert(d_memory[req_id] == NULL);
            break;
        case GC:
            //assert(d_memory[req_id] == NULL);
            break;
        default:
            printf("error\n");
            break;
    }
    requests->request_signal[thid] = request_empty;
    release_semaphore((int*)requests->lock, thid);
    __threadfence();
    // SEMAPHORE
}

__device__
void Runtime::request_async(request_type type,
                            volatile int** dest,
                            int size_to_alloc = 0){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // wait for request to be posted
    while (is_working()){
        if (is_available(thid)){
            post_request(type, size_to_alloc);
            break;
        }
        __threadfence();
    }
    __threadfence();
    // do not wait request to be completed
}

__device__
void Runtime::request(request_type type,
                      volatile int** dest,
                      int size_to_alloc = 0){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // wait for request to be posted
    while (is_working()){
        if (is_available(thid)){
            post_request(type, size_to_alloc);
            break;
        }
        __threadfence();
    }
    __threadfence();

    // wait for request to be completed
    while (is_working()){
        if (is_finished(thid)){
            request_processed(type, dest);
            break;
        }
        __threadfence();
    }
    __threadfence();
}

__device__
void Runtime::_request_processing(int request_id, MemoryManagerType* mm){
    
    debug("request processing!\n");

    // SEMAPHORE
    acquire_semaphore((int*)(requests->lock), request_id);
    debug("MEM MANAGER %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, request_id, blockIdx.x, warp_id(), lane_id(), sm_id());
    
    auto addr_id = requests->request_id[request_id];
    int request_status;
    
    switch (requests->request_signal[request_id]){
    //switch (type(request_id)){
        case MALLOC:
            if (addr_id == -1){
                //addr_id = atomicAdd((int*)&requests.request_counter[0], 1);
                addr_id = atomicAdd((int*)&(requests->request_counter[0]), 1);
                //requests.request_id[request_id] = addr_id;
                requests->request_id[request_id] = addr_id;
            }else{
                printf("addr_id %d\n", addr_id);
               // assert(requests.d_memory[addr_id] == NULL);
                assert(requests->d_memory[addr_id] == NULL);
            }
            __threadfence();
            //requests.d_memory[addr_id] = reinterpret_cast<volatile int*>
            requests->d_memory[addr_id] = reinterpret_cast<volatile int*>
                //(mm->malloc(4+requests.request_mem_size[request_id]));
                (mm->malloc(4+requests->request_mem_size[request_id]));
            __threadfence();
            //assert(requests.d_memory[addr_id]);
            assert(requests->d_memory[addr_id]);
            //requests.d_memory[addr_id][0] = 0;
            requests->d_memory[addr_id][0] = 0;
            //requests.request_dest[request_id] = &requests.d_memory[addr_id][1];
            requests->request_dest[request_id] = &(requests->d_memory[addr_id][1]);
            //atomicExch((int*)&requests.request_signal[request_id], request_done);
            atomicExch((int*)&(requests->request_signal[request_id]), request_done);
            //if (requests.d_memory[addr_id][0] != 0)
            if (requests->d_memory[addr_id][0] != 0)
                //printf("d_memory{%d} = %d\n", addr_id, requests.d_memory[addr_id][0]);
                printf("d_memory{%d} = %d\n", addr_id, requests->d_memory[addr_id][0]);
            //assert(requests.d_memory[addr_id][0] == 0);
            assert(requests->d_memory[addr_id][0] == 0);
            __threadfence();
            break;

        case FREE:
            //assert(requests.d_memory[addr_id]);
            assert(requests->d_memory[addr_id]);
            //if (requests.d_memory[addr_id][0] != 0)
            if (requests->d_memory[addr_id][0] != 0)
                //printf("d_memory{%d} = %d\n", addr_id, requests.d_memory[addr_id][0]);
                printf("d_memory{%d} = %d\n", addr_id, requests->d_memory[addr_id][0]);
            //assert(requests.d_memory[addr_id][0] == 0);
            assert(requests->d_memory[addr_id][0] == 0);
            //request_status = requests.d_memory[addr_id][0] - 1;
            request_status = requests->d_memory[addr_id][0] - 1;
            //requests.d_memory[addr_id][0] -= 1;
            requests->d_memory[addr_id][0] -= 1;
            //requests.request_dest[request_id] = NULL;
            requests->request_dest[request_id] = NULL;
            //assert(requests.d_memory[addr_id][0] == -1);
            assert(requests->d_memory[addr_id][0] == -1);
            if (request_status < 0){
                //atomicExch((int*)&requests.request_signal[request_id], request_gc);
                atomicExch((int*)&(requests->request_signal[request_id]), request_gc);
            }else{
                assert(1);
                printf("should not be here!\n");
                //atomicExch((int*)&requests.request_signal[request_id], request_done);
                atomicExch((int*)&requests->request_signal[request_id], request_done);
            }
            break;

        case GC:
            //assert(requests.d_memory[addr_id]);
            //assert(requests.d_memory[addr_id][0] == -1);
            assert(requests->d_memory[addr_id]);
            assert(requests->d_memory[addr_id][0] == -1);
            __threadfence();
            //mm->free((void*)requests.d_memory[addr_id]);
            mm->free((void*)requests->d_memory[addr_id]);
            __threadfence();
            //requests.d_memory[addr_id] = NULL;
            requests->d_memory[addr_id] = NULL;
            //atomicExch((int*)&requests.request_signal[request_id], request_done);
            atomicExch((int*)&requests->request_signal[request_id], request_done);
            break;

        default:
            printf("request processing fail\n");

    }
    //release_semaphore((int*)requests.lock, request_id);
    release_semaphore((int*)(requests->lock), request_id);
    // SEMAPHORE
}
__device__
void Runtime::malloc(volatile int** ptr, size_t size){
    request((request_type)MALLOC, ptr, size);
}

__device__
void Runtime::malloc_async(volatile int** ptr, size_t size){
    request_async((request_type)MALLOC, ptr, size);
}

__device__
void Runtime::free(volatile int* ptr){
    request((request_type)FREE, &ptr, 0);
}

__device__
void Runtime::free_async(volatile int** ptr){
    request_async((request_type)FREE, ptr, 0);
}
/*
    Size - the number of threads assigned to the application
           One reqeust per thread at a time
 */
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
    int* h_memory = (int*)std::malloc(requests_num* sizeof(int));
    GUARD_CU(cudaMemcpy(h_memory, d_memory, sizeof(int)*requests_num, cudaMemcpyDeviceToHost));
    GUARD_CU(cudaStreamSynchronize(0));
    GUARD_CU(cudaPeekAtLastError());
}

__global__
void test2(volatile int** d_memory, int size){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid < size){
        assert(d_memory[thid] == NULL);
    }
}*/
}
#endif
