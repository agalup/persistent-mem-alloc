#include <iostream>
#include <string>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>
#include <any>

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "PerformanceMeasure.cuh"
#include "Utility.h"
#include "cuda.h"
#include "pmm-utils.cuh"
#include "pmm-utils.h"

using namespace std;

// Launch tests
void mps_monolithic_app(int, int, int, size_t*, size_t, int, int*, int*, 
                        int*, int*, float*, int*, int, int, int);

void simple_monolithic_app(int, int, int, size_t*, size_t, int, int*, 
                           int*, int*, int*, float*, int*, int, int);

void mps_app(int, int, int, size_t*, size_t, int, int*, int*, int*, 
             int*, float*, int*, int, int, int);

// Servies
// Memory Manager Service
void start_memory_manager(PerfMeasure&, uint32_t, uint32_t, CUcontext&, Runtime&);
__global__ void mem_manager(Runtime);

// Garbage Collector Service
void start_garbage_collector(PerfMeasure&, uint32_t, uint32_t, CUcontext&, Runtime&);
__global__ void garbage_collector(Runtime);

// Callback Service 
void start_callback_server(PerfMeasure&, uint32_t, uint32_t, CUcontext&, Runtime&);

// TODO: Queue Service TODO
// TODO: Fency mechanism of turning on/off services from Runtime.
// TODO: Testing without garbage collector.

// Clean up
void clean_memory(uint32_t, uint32_t, Runtime&);

__global__ void mem_free(Runtime);

// TESTS
void start_application(PerfMeasure&, uint32_t, uint32_t, volatile int*, volatile int*, 
                        int*, int*, int, bool&, Runtime&);

__global__ void app_test(int*, int*, int, Runtime);
__global__ void mono_app_test(volatile int**, volatile int*, int*, int*, MemoryManagerType*);

__global__ void app_async_request_test(int*, int*, int, Runtime);

__global__ void app_one_per_warp_test(int*, int*, int, Runtime);

__global__ void app_async_one_per_warp_test(int*, int*, int, Runtime);

__global__ void app_one_per_block_test(int*, int*, int, Runtime);

__global__ void app_async_one_per_block_test(int*, int*, int, Runtime);

__global__ void callback_test(int*, Runtime);

/* TODO: write down the pmm_init arguments:
 * INPUT
 *  mono
 *  kernel_iteration_num
 *  size_to_alloc
 *  ins_size
 *  num_iterations
 *  SMs
 * OUTPUT
 *  sm_app
 *  sm_mm
 *  sm_gc
 *  allocs
 *  uni_req_per_sec
 *  array_size
 */
// TODO test of memory allocation of random sizes within a kernel
extern "C" void pmm_init(int mono, int kernel_iteration_num, int size_to_alloc, 
        size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
        int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
        int* array_size, int block_size, int device, int cb_number){

    GUARD_CU(cudaSetDevice(device));

    printf("mono %d\n", mono);
    printf("device %d\n", device);

    if (mono == MPS_mono){
        printf("MPS_mono\n");

        mps_monolithic_app(mono, kernel_iteration_num, 
                size_to_alloc, ins_size, num_iterations, 
                SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device, cb_number);

        printf("MPS_mono\n");
    }else if (mono == simple_mono){
        printf("simple mono\n");

        simple_monolithic_app(mono, kernel_iteration_num, 
                size_to_alloc, ins_size, num_iterations, 
                SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device);

        printf("simple mono\n");
    }else{
        printf("other mono\n");
        mps_app(mono, kernel_iteration_num, size_to_alloc, ins_size, 
                num_iterations, SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device, cb_number);
    }
}

__global__
void mem_free(Runtime runtime){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= runtime.app_threads_num){
        return;
    }
    __threadfence();
    if (runtime.requests->d_memory[thid]){
        printf("sync error: %d was not released before\n", thid);
#ifdef OUROBOROS__
        runtime.mem_manager->free((void*)runtime.requests->d_memory[thid]);
#else
        //GUARD_CU_DEV(cudaFree((void*)runtime.requests->d_memory[thid]));
        //cudaFree((void*)runtime.requests->d_memory[thid]);
#endif
    }
}


__global__
void garbage_collector(Runtime runtime){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    runtime.gc->start();
    RequestType* reqs = runtime.requests;
    while (runtime.is_working()){
        //debug("hello gc! %d\n", thid);
        for (int request_id = thid; !runtime.exit_signal[0] && 
                request_id < runtime.size();
                request_id += blockDim.x*gridDim.x){
            __threadfence();
            if (reqs->type(request_id) == GC){
                runtime._request_processing(request_id);
                __threadfence();
            }
        }
        __threadfence();
    }
}


__global__
void mem_manager(Runtime runtime){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    runtime.mm->start();
    RequestType* reqs = runtime.requests;
    while (runtime.is_working()){
        //debug("hello mm %d, request no %d!\n", thid, reqs->number());
        for (int request_id = thid; !runtime.exit_signal[0] &&
                request_id < runtime.size();
                request_id += blockDim.x*gridDim.x){
            __threadfence();
            if (reqs->type(request_id) == MALLOC or reqs->type(request_id) == FREE){
                runtime._request_processing(request_id);
                __threadfence();
                debug("mm: request done %d\n", request_id);
            }
        }
        __threadfence();
    }
}

// TESTS

__global__
void mono_app_test(volatile int** d_memory, 
                   volatile int* exit_counter, 
                   int* size_to_alloc,
                   int* iter_num,
                   MemoryManagerType* mm){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    assert(d_memory);
    assert(exit_counter);
    assert(size_to_alloc);
    assert(iter_num);

    for (int i=0; i<iter_num[0]; ++i){
        __threadfence();

        volatile int* new_ptr = NULL;

#ifdef OUROBOROS__
        d_memory[thid] = reinterpret_cast<volatile int*>(mm->malloc(4+size_to_alloc[0])); 
#else
        GUARD_CU_DEV(cudaMalloc((void**)&d_memory[thid], 4+size_to_alloc[0]));
#endif
        d_memory[thid][0] = 0;
        new_ptr = &d_memory[thid][1];
        new_ptr[0] = thid;

        __threadfence();

        assert(d_memory[thid]);
        //int value = d_memory[thid][0];
        //if (value != 0) printf("val = %d\n", value);
        //value = d_memory[thid][1];
        assert(new_ptr[0] == thid);

        __threadfence();

#ifdef OUROBOROS__
        mm->free((void*)d_memory[thid]);
#else
        //GUARD_CU_DEV(cudaFree((void*)d_memory[thid]));
#endif
        __threadfence();
        d_memory[thid] = NULL;

        __threadfence();
    }

    atomicAdd((int*)&exit_counter[0], 1);
    __threadfence();
}


__global__
void app_one_per_block_test(int* size_to_alloc,
                            int* iter_num,
                            int MONO,
                            Runtime runtime){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int* ptr_tab;
    
    for (int i=0; i<iter_num[0]; ++i){
        // ALLOCTAION
        volatile int* new_ptr = NULL;
        runtime.malloc_block((volatile int**)&new_ptr, (volatile int**)&ptr_tab, size_to_alloc[0]);
        assert(new_ptr);

        // WRITE
        new_ptr[0] = thid;

        // READ
        assert(new_ptr[0] == thid);
        
        // RECLAMATION
        runtime.free_block(new_ptr);
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}


__global__
void app_one_per_warp_test(int* size_to_alloc,
                           int* iter_num,
                           int MONO,
                           Runtime runtime){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x/32;
    __shared__ int* ptr_tab[32];
    
    for (int i=0; i<iter_num[0]; ++i){
        // ALLOCTAION
        volatile int* new_ptr = NULL;
        runtime.malloc_warp((volatile int**)&new_ptr, (volatile int**)&ptr_tab[warp_id], size_to_alloc[0]);
        assert(new_ptr);

        // WRITE
        new_ptr[0] = thid;

        // READ
        assert(new_ptr[0] == thid);
        
        // RECLAMATION
        runtime.free_warp(new_ptr);
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}


__global__
void app_async_one_per_block_test(int* size_to_alloc,
                                  int* iter_num,
                                  int MONO,
                                  Runtime runtime){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ Future future_tmp;

    for (int i=0; i<iter_num[0]; ++i){
        // ALLOC
        runtime.malloc_block_async(future_tmp, size_to_alloc[0]);
        volatile int* new_ptr = future_tmp.get_block_async(size_to_alloc[0]);
        __threadfence();
        assert(new_ptr);

        // WRITE
        new_ptr[0] = thid;
        __threadfence();

        // READ
        assert(new_ptr[0] == thid);
       
        // RECLAMATION
        runtime.free_block_async(future_tmp);
        if (threadIdx.x == 0)
            runtime.wait((request_type)FREE, thid, &(future_tmp.ptr));
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}


__global__
void app_async_one_per_warp_test(int* size_to_alloc,
                                 int* iter_num,
                                 int MONO,
                                 Runtime runtime){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x/32;
    __shared__ Future future_tmp[32];

    for (int i=0; i<iter_num[0]; ++i){
        // ALLOC
        runtime.malloc_warp_async(future_tmp[warp_id], size_to_alloc[0]);
        volatile int* new_ptr = future_tmp[warp_id].get_warp_async(size_to_alloc[0]);
        __threadfence();
        assert(new_ptr);

        // WRITE
        new_ptr[0] = thid;
        __threadfence();

        // READ
        assert(new_ptr[0] == thid);
       
        // RECLAMATION
        runtime.free_warp_async(future_tmp[warp_id]);
        if (threadIdx.x%32 == 0)
            runtime.wait((request_type)FREE, thid, &(future_tmp[warp_id].ptr));
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}


__global__
void app_async_request_test(int* size_to_alloc,
                            int* iter_num,
                            int MONO,
                            Runtime runtime){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i=0; i<iter_num[0]; ++i){
        // MEMORY ALLOCATION
        Future future_ptr; //Can I pass it outside of the kernel?
        runtime.malloc_async(future_ptr, size_to_alloc[0]);
        
        // WRITE TEST
        volatile int* new_ptr = future_ptr.get();
        __threadfence();
        assert(new_ptr);
        new_ptr[0] = thid;

        // READ TEST
        assert(new_ptr[0] == thid);

        // MEMORY RECLAMATION
        //__threadfence();
        runtime.free_async(future_ptr);
        runtime.wait((request_type)FREE, thid, &future_ptr.ptr);
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}

__global__
void callback_async_test(int* iter_num, Runtime runtime){
    //int thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i=0; i<iter_num[0]; ++i){
        // CALLBACK
        //volatile int* new_ptr;
        Future complete;
        runtime.callback_async(complete, 1);
        runtime.callback_async(complete, 2);
        debug("callback done [%s:%d]\n", __FILE__, __LINE__);

        //complete.cb_get(); hangs because one thread on CPU is a bottleneck
        __threadfence();
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}

__global__
void callback_test(int* iter_num, Runtime runtime){
    for (int i=0; i<iter_num[0]; ++i){
        // CALLBACK
        volatile int* new_ptr;
        runtime.callback(&new_ptr, 0);
        debug("callback done [%s:%d]\n", __FILE__, __LINE__);
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}

__global__
void app_test(int* size_to_alloc, int* iter_num,
              int MONO, Runtime runtime){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i=0; i<iter_num[0]; ++i){
        // ALLOC
        volatile int* new_ptr;
        runtime.malloc(&new_ptr, size_to_alloc[0]);
        assert(new_ptr);

        // WRITE
        new_ptr[0] = thid;

        // READ
        assert(new_ptr[0] == thid);

        // RECLAMATION
        runtime.free(new_ptr);
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}


//TODO: put into runtime class
void start_memory_manager(PerfMeasure& timing_mm, 
                          uint32_t mm_grid_size,
                          uint32_t block_size, 
                          CUcontext& mm_ctx,
                          Runtime& runtime){
    timing_mm.startMeasurement();
    
    void *args[] = {&runtime};
    GUARD_CU(cudaLaunchCooperativeKernel((void*)mem_manager, mm_grid_size, block_size, args));
    //GUARD_CU(cudaLaunchKernel((void*)mem_manager, mm_grid_size, block_size, args));
    GUARD_CU((cudaError_t)cudaGetLastError());
    GUARD_CU(cudaPeekAtLastError());

    timing_mm.stopMeasurement();
}

void start_garbage_collector(PerfMeasure& timing_gc, 
                             uint32_t gc_grid_size,
                             uint32_t block_size, 
                             CUcontext& gc_ctx,
                             Runtime& runtime){
    timing_gc.startMeasurement();
    
    void *args[] = {&runtime};
    GUARD_CU(cudaLaunchCooperativeKernel((void*)garbage_collector, gc_grid_size, block_size, args));
    //GUARD_CU(cudaLaunchKernel((void*)garbage_collector, gc_grid_size, block_size, args));
    GUARD_CU((cudaError_t)cudaGetLastError());
    GUARD_CU(cudaPeekAtLastError());
   
    timing_gc.stopMeasurement();
}

void start_callback_server(PerfMeasure& timing_cb, 
                          uint32_t cb_grid_size,
                          uint32_t block_size, 
                          CUcontext& cb_ctx,
                          Runtime& runtime){
    Callback_fn fn1 = [](int* ptr){
        printf("Registered callback\n");
        };
    Callback_fn fn2 = [](int* ptr){
        //printf("cudaMalloc within HOST\n");
        GUARD_CU(cudaMalloc((void**)&ptr, sizeof(int)));
        };
    Callback_fn fn3 = [](int* ptr){
        //printf("cudaFree within HOST\n");
        GUARD_CU(cudaFree(ptr));
        };
    runtime.register_cb(fn1);
    runtime.register_cb(fn2);
   // runtime.register_cb(fn3);
    timing_cb.startMeasurement();
    runtime.cb->start();
    while (runtime.is_working()){
        for (int i = 0; i < runtime.app_threads_num; ++i){
            //printf("for thread %d\n", i);
            if (runtime.there_is_a_callback(i)){
                int cb_id = runtime.callback_id(i);
                debug("callback by %d of id %d\n", i, cb_id);
                runtime.callback_run(cb_id)(NULL);
                runtime.callback_close(i);
                fflush(stdout);
            }
        }
    }
    GUARD_CU((cudaError_t)cudaGetLastError());
    GUARD_CU(cudaPeekAtLastError());
    timing_cb.stopMeasurement();
}

void clean_memory(uint32_t grid_size,
                  uint32_t block_size, 
                  Runtime& runtime){

    runtime.exit_signal[0] = 1;
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    void* args[] = {&runtime};
    GUARD_CU(cudaLaunchCooperativeKernel((void*)mem_free, grid_size, block_size, args));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
    
    runtime.free();
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}


void start_application(PerfMeasure& timing_sync, 
                       uint32_t grid_size,
                       uint32_t block_size, 
                       volatile int* exit_signal,
                       volatile int* exit_counter,
                       int* dev_size_to_alloc, 
                       int* dev_iter_num,
                       int mono, 
                       bool& kernel_complete,
                       Runtime& runtime){
    auto dev_mm = runtime.mem_manager;
    if (mono == MPS_mono){
        void* args[] = {&(runtime.requests->d_memory), &exit_counter, &dev_size_to_alloc, &dev_iter_num, &dev_mm};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        debug("start application: MPS mono!\n");
        //GUARD_CU(cudaLaunchKernel((void*)mono_app_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)mono_app_test, grid_size, block_size, args, 0, 0));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        debug("ctx sync done!\n");
        fflush(stdout);
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == one_per_warp){
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        debug("start application one per warp\n");
        //GUARD_CU(cudaLaunchKernel((void*)app_one_per_warp_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_one_per_warp_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        debug("stop application one per warp\n");
        fflush(stdout);
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == async_request){
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        //GUARD_CU(cudaLaunchKernel((void*)app_async_request_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_async_request_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == one_per_block){
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_one_per_block_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == async_one_per_warp){
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_async_one_per_warp_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == async_one_per_block){
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_async_one_per_block_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == callback_type){
        void* args[] = {&dev_iter_num, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        GUARD_CU(cudaLaunchCooperativeKernel((void*)callback_async_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }else{
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        //GUARD_CU(cudaLaunchKernel((void*)app_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }
    kernel_complete = true;
}

void simple_monolithic_app(int mono, int kernel_iteration_num, 
            int size_to_alloc, size_t* ins_size, size_t num_iterations, 
            int SMs, int* sm_app, int* sm_mm, int* sm_gc, int* allocs, 
            float* uni_req_per_sec, int* array_size, int block_size, 
            int device_id){

    CUcontext default_ctx;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));

#ifdef OUROBOROS__
    //Ouroboros initialization
    auto instant_size = *ins_size;
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);
#endif

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    
    volatile int* exit_signal;
    volatile int* exit_counter;
    int* dev_size_to_alloc;
    int* dev_kernel_iteration_num;
    
    GUARD_CU(cudaMallocManaged(&exit_signal, sizeof(int32_t)));
    GUARD_CU(cudaMallocManaged(&exit_counter, sizeof(uint32_t)));
    GUARD_CU(cudaMallocManaged(&dev_size_to_alloc, sizeof(int)));
    GUARD_CU(cudaMallocManaged(&dev_kernel_iteration_num, sizeof(int)));
   
    *dev_size_to_alloc = size_to_alloc;
    *dev_kernel_iteration_num = kernel_iteration_num;
    
    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, device_id));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_size_to_alloc, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_kernel_iteration_num, sizeof(int), device, NULL));

    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());

    int app_grid_size = SMs;
    int requests_num = app_grid_size * block_size;
    
    sm_app[0] = app_grid_size;
    sm_mm[0] = 0;
    sm_gc[0] = 0;
    allocs[0] = requests_num;

    cudaStream_t app_stream;
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());

    PerfMeasure malloc_total_sync;
    for (int iteration = 0; iteration < num_iterations; ++iteration){
        *exit_counter = 0;
        volatile int** d_memory{nullptr};
        GUARD_CU(cudaMalloc(&d_memory, requests_num * sizeof(volatile int*)));
        GUARD_CU(cudaPeekAtLastError());
        malloc_total_sync.startMeasurement();

#ifdef OUROBOROS__
        auto dev_mm = memory_manager.getDeviceMemoryManager();
#else
        void** dev_mm = NULL;
#endif
        mono_app_test<<<app_grid_size, block_size, 0, app_stream>>>(
                        d_memory, exit_counter, dev_size_to_alloc, 
                        dev_kernel_iteration_num, dev_mm); 

        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        malloc_total_sync.stopMeasurement();
        GUARD_CU(cudaFree(d_memory));
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
    }

    auto malloc_total_sync_res = malloc_total_sync.generateResult();
    auto total_iters = kernel_iteration_num*num_iterations;
    //uni_req_per_sec[0] = (requests_num * 1000.0)/(malloc_total_sync_res.mean_/total_iters);
    uni_req_per_sec[0] = (requests_num * 2000.0)/malloc_total_sync_res.mean_;

    printf("#measurements %d, mean %.2lf, #total iters %lu (host: %lu)\n", malloc_total_sync_res.num_,
    malloc_total_sync_res.mean_, total_iters, num_iterations);

    printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
            app_grid_size, 0, 0, uni_req_per_sec[0]);

    *array_size = 1;

}

__host__
void mps_monolithic_app(int mono, int kernel_iteration_num, int size_to_alloc, 
            size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
            int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
            int* array_size, int block_size, int device_id, int cb_number = 0){

    //Ouroboros initialization
    MemoryManagerType memory_manager;
#ifdef OUROBOROS__
    auto instant_size = *ins_size;
    memory_manager.initialize(instant_size);
#else
#endif

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    
    volatile int* exit_signal;
    volatile int* exit_counter;
    GUARD_CU(cudaMallocManaged(&exit_signal, sizeof(int32_t)));
    GUARD_CU(cudaMallocManaged(&exit_counter, sizeof(uint32_t)));

    CUcontext app_ctx;
    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, device_id));
    GUARD_CU((cudaError_t)cuCtxCreate(&app_ctx, 0, device));

    CUcontext default_ctx;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));

    int* dev_size_to_alloc;
    int* dev_kernel_iteration_num;
    GUARD_CU(cudaMallocManaged(&dev_size_to_alloc, sizeof(int)));
    GUARD_CU(cudaMallocManaged(&dev_kernel_iteration_num, sizeof(int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
   
    *dev_size_to_alloc = size_to_alloc;
    *dev_kernel_iteration_num = kernel_iteration_num;
    
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)dev_size_to_alloc, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)dev_kernel_iteration_num, sizeof(int), device, NULL));

    int app_grid_size = SMs;
    int requests_num = app_grid_size * block_size;
    sm_app[0] = app_grid_size;
    sm_mm[0] = 0;
    sm_gc[0] = 0;
    allocs[0] = requests_num;

    PerfMeasure malloc_total_sync;

    for (int iteration = 0; iteration < num_iterations; ++iteration){
        //printf("iter %d, requests_num %d\n", iteration, requests_num);

        *exit_signal = 0;
        *exit_counter = 0;

        GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_signal, sizeof(int), device, NULL));
        GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_counter, sizeof(int), device, NULL));

        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());

        Runtime runtime;
        runtime.init(requests_num, device, memory_manager, cb_number);

        // Run APP (all threads do malloc)
        bool kernel_complete = false;
        std::thread app_thread{[&] {
            GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
            //GUARD_CU((cudaError_t)cuCtxSynchronize());
            debug("start app\n");
            //malloc_total_sync.startMeasurement();

            //mps_monolithic_app
            start_application(malloc_total_sync, app_grid_size, 
                    block_size, exit_signal, exit_counter, 
                    dev_size_to_alloc, dev_kernel_iteration_num, 
                    mono, kernel_complete, runtime);
            debug("app done, sync\n");
            GUARD_CU((cudaError_t)cuCtxSynchronize());
            //malloc_total_sync.stopMeasurement();
            GUARD_CU(cudaPeekAtLastError());
            debug("done\n");
        }};

        debug("join app\n");
        app_thread.join();
        debug("app joined\n");

        if (not kernel_complete){
            printf("kernel is not completed, free memory which app allocated\n");
            clean_memory(app_grid_size, block_size, runtime);
            continue;
        }

        *exit_signal = 1;

        clean_memory(app_grid_size, block_size, runtime);
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
    }

    GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));
    //GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    auto malloc_total_sync_res = malloc_total_sync.generateResult();
    auto total_iters = kernel_iteration_num*num_iterations;
    //uni_req_per_sec[0] = (requests_num * 1000.0)/(malloc_total_sync_res.mean_/total_iters);
    uni_req_per_sec[0] = (requests_num * 2000.0)/malloc_total_sync_res.mean_;

    printf("#measurements %d, mean %.2lf, #total iters %lu (host: %lu)\n", malloc_total_sync_res.num_,
    malloc_total_sync_res.mean_, total_iters, num_iterations);

    printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
            app_grid_size, 0, 0, uni_req_per_sec[0]);

    *array_size = 1;

}

__host__
void mps_app(int mono, int kernel_iteration_num, int size_to_alloc, 
             size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
             int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
             int* array_size, int block_size, int device_id, int cb_number){

    CUcontext default_ctx;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));

    //Ouroboros initialization
    MemoryManagerType memory_manager;
#ifdef OUROBOROS__
    auto instant_size = *ins_size;
    memory_manager.initialize(instant_size);
#endif

    int total_gpus = 0;
    GUARD_CU((cudaError_t)cudaGetDeviceCount(&total_gpus));
    debug("device %d/%d\n", device_id, total_gpus);
    debug("Block size = %d\n", block_size);
    
    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, device_id));

    int* dev_size_to_alloc;        
    int* dev_kernel_iteration_num; 

    GUARD_CU(cudaMallocManaged(&dev_size_to_alloc, sizeof(int)));
    GUARD_CU(cudaMallocManaged(&dev_kernel_iteration_num, sizeof(int)));

    *dev_size_to_alloc = size_to_alloc;
    *dev_kernel_iteration_num = kernel_iteration_num;
    
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_size_to_alloc, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_kernel_iteration_num, sizeof(int), device, NULL));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    int it = 0;
    int cb_grid_size = 1;
    for (int app_grid_size=55; app_grid_size<SMs-cb_grid_size; ++app_grid_size){
        for (int mm_grid_size=1; mm_grid_size<(SMs-app_grid_size-cb_grid_size); ++mm_grid_size){
            int gc_grid_size = SMs - app_grid_size - mm_grid_size - cb_grid_size;
            if (gc_grid_size < 1) continue;

            int requests_num{app_grid_size * block_size};
            //printf("SMs: app %d, mm %d, gc %d, cb %d, total %d\n", 
            //        app_grid_size, mm_grid_size, gc_grid_size, cb_grid_size, SMs);
            //printf("requests_num %d\n", requests_num);
            //fflush(stdout);

            //output
            sm_app[it] = app_grid_size;
            sm_mm [it] = mm_grid_size;
            sm_gc [it] = gc_grid_size;
            allocs[it] = requests_num;

            int app_numBlocksPerSm = 1;
            int gc_numBlocksPerSm =  1;
            int mm_numBlocksPerSm =  1;
            int cb_numBlocksPerSm =  1;

            debug("num blocks per sm by cudaOccMaxActBlPerSM: app %d, mm %d, gc %d\n", 
                                    app_numBlocksPerSm, gc_numBlocksPerSm, mm_numBlocksPerSm);
            //fflush(stdout);

            CUexecAffinityParam_v1 app_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                (unsigned int)app_grid_size * app_numBlocksPerSm};
            CUexecAffinityParam_v1 mm_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                (unsigned int)mm_grid_size  *  mm_numBlocksPerSm};
            CUexecAffinityParam_v1 gc_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                (unsigned int)gc_grid_size  *  gc_numBlocksPerSm};
            CUexecAffinityParam_v1 cb_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                (unsigned int)cb_grid_size  *  cb_numBlocksPerSm};

            auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
            //auto affinity_flags = CUctx_flags::CU_CTX_SCHED_SPIN;
            //auto affinity_flags = CUctx_flags::CU_CTX_SCHED_YIELD;
            //auto affinity_flags = CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC;
            //auto affinity_flags = CUctx_flags::CU_CTX_BLOCKING_SYNC;
            //auto affinity_flags = CUctx_flags::CU_CTX_MAP_HOST;
            //auto affinity_flags = CUctx_flags::CU_CTX_LMEM_RESIZE_TO_MAX;

            CUcontext app_ctx, mm_ctx, gc_ctx, cb_ctx;
            GUARD_CU((cudaError_t)cuCtxCreate_v3(
                        &app_ctx, &app_param, 1, affinity_flags, device));

            GUARD_CU((cudaError_t)cuCtxCreate_v3(
                        &mm_ctx, &mm_param, 1, affinity_flags, device));

            GUARD_CU((cudaError_t)cuCtxCreate_v3(
                        &gc_ctx, &gc_param, 1, affinity_flags, device));

            GUARD_CU((cudaError_t)cuCtxCreate_v3(
                        &cb_ctx, &cb_param, 1, affinity_flags, device));

            GUARD_CU(cudaPeekAtLastError());
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU((cudaError_t)cudaGetLastError());

            //Timing variables
            PerfMeasure malloc_total_sync, timing_mm, timing_gc, timing_cb;
            for (int iteration = 0; iteration < num_iterations; ++iteration){
                debug("iteration %d/%d\n", iteration, num_iterations);
                
                Runtime runtime;
                runtime.init(requests_num, device, memory_manager, cb_number);

                debug("start threads\n");
                // Run Memory Manager (Presistent kernel)
                std::thread mm_thread{[&] {
                    GUARD_CU((cudaError_t)cuCtxSetCurrent(mm_ctx));
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("start mm\n");
                    start_memory_manager(timing_mm, mm_numBlocksPerSm*mm_grid_size, 
                                         block_size, mm_ctx, runtime);
                    debug("mm done, sync\n");
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("done\n");
                }};
                //std::this_thread::sleep_for(std::chrono::seconds(1));

                // Run Garbage Collector (persistent kernel)
                std::thread gc_thread{[&] {
                    GUARD_CU((cudaError_t)cuCtxSetCurrent(gc_ctx));
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("start gc\n");
                    start_garbage_collector(timing_gc, gc_numBlocksPerSm*gc_grid_size, 
                                            block_size, gc_ctx, runtime);
                    debug("gc done, sync\n");
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("done\n");
                }}; 
                //std::this_thread::sleep_for(std::chrono::seconds(1));

                // Callback Server (persistent kernel)
                std::thread cb_thread{[&] {
                    GUARD_CU((cudaError_t)cuCtxSetCurrent(cb_ctx));
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("start callback\n");
                    start_callback_server(timing_cb, cb_numBlocksPerSm*cb_grid_size, 
                                          block_size, cb_ctx, runtime);
                    debug("callback done, sync\n");
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("done\n");
                }};
                //std::this_thread::sleep_for(std::chrono::seconds(1));

                //printf("-");
                fflush(stdout);
                while (!(runtime.gc->is_running() && 
                         runtime.mm->is_running() && 
                         runtime.cb->is_running()));

                GUARD_CU((cudaError_t)cudaGetLastError());

                debug("app_numBlocksPerSm %d, app_grid_size %d, block_size %d\n", 
                                    app_numBlocksPerSm, app_grid_size, block_size);
                fflush(stdout);

                // Run APP (all threads do malloc)
                bool kernel_complete = false;
                std::thread app_thread{[&] {
                    GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    GUARD_CU((cudaError_t)cudaGetLastError());
                    debug("start app\n");
                    //mps_app
                    start_application(malloc_total_sync, 
                            app_numBlocksPerSm*app_grid_size, 
                            block_size, runtime.exit_signal, 
                            runtime.exit_counter, dev_size_to_alloc, 
                            dev_kernel_iteration_num, mono, 
                            kernel_complete, runtime);
                    debug("done before ctx sync\n");
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("done\n");
                }};

                //std::this_thread::sleep_for(std::chrono::seconds(1));

                debug("join app\n");
                app_thread.join();
                debug("app joined\n");

                if (not kernel_complete){
                    printf("kernel is not completed, free memory which app allocated\n");
                    clean_memory(app_grid_size, block_size, runtime);
                    continue;
                }

                runtime.stop();

                debug("join mm\n");
                mm_thread.join();
                debug("mm joined\n");

                debug("join gc\n");
                gc_thread.join();
                debug("gc joined\n");

                debug("join callback\n");
                cb_thread.join();
                debug("callback joined\n");

                //Deallocate device memory
                cuCtxSetCurrent(default_ctx);
                GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
                clean_memory(app_grid_size, block_size, runtime);

                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());
            }
            debug("\n");
            debug("done\n");

            GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));
            GUARD_CU((cudaError_t)cuCtxDestroy(gc_ctx));
            GUARD_CU((cudaError_t)cuCtxDestroy(mm_ctx));
            GUARD_CU((cudaError_t)cuCtxDestroy(cb_ctx));
            GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());

            // Output: the number of requests done per a second
            auto malloc_total_sync_res = malloc_total_sync.generateResult();
            auto total_iters = kernel_iteration_num*num_iterations;
            //uni_req_per_sec[it] = (requests_num * 1000.0)/(malloc_total_sync_res.mean_/total_iters);
            uni_req_per_sec[it] = (requests_num * 2000.0)/malloc_total_sync_res.mean_;

            switch (mono){
                case MPS_mono:
                    printf("MPS mono. ");
                    break;
                case simple_mono:
                    printf("Simple mono. ");
                    break;
                case one_per_warp:
                    printf("One per warp. "); 
                    break;
                case one_per_block:
                    printf("One per block.");
                    break;
                case async_request:
                    printf("Async request. "); 
                    break;
                case async_one_per_warp:
                    printf("Async one per warp. ");
                    break;
                case async_one_per_block:
                    printf("Async one per block.");
                    break;
                case callback_type:
                    printf("Callback. ");
                    break;
                default:
                    printf("MPS service. "); 
                    break;
            }

            printf("#measurements %d, mean %.2lf, #total iters %lu (by host: %lu)\n", 
                    malloc_total_sync_res.num_, 
                    malloc_total_sync_res.mean_, total_iters, num_iterations);

            printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", 
                    requests_num, app_grid_size, mm_grid_size, 
                    gc_grid_size, uni_req_per_sec[it]);
            ++it;
        }
    }
    *array_size = it;
}

