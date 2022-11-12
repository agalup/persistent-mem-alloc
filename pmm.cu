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

using namespace std;

extern "C" {

__global__
void mem_free(Runtime runtime){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= runtime.app_threads_num){
        return;
    }
    __threadfence();
    if (runtime.requests->d_memory[thid]){
        printf("sync error: %d was not released before\n", thid);
        runtime.mem_manager->free((void*)runtime.requests->d_memory[thid]);
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

//producer
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
//consumer
__global__
void mono_app_test(
        volatile int** d_memory, 
        volatile int* exit_counter, 
        int* size_to_alloc,
        int* iter_num,
        MemoryManagerType* mm){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if(thid == 0)
        printf("d_memory %x\n", d_memory);

    assert(d_memory);
    assert(exit_counter);
    assert(size_to_alloc);
    assert(iter_num);

    for (int i=0; i<iter_num[0]; ++i){
        __threadfence();

        volatile int* new_ptr = NULL;

        d_memory[thid] = reinterpret_cast<volatile int*>(mm->malloc(4+size_to_alloc[0])); 
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

        mm->free((void*)d_memory[thid]);
        __threadfence();
        d_memory[thid] = NULL;

        __threadfence();
    }

    atomicAdd((int*)&exit_counter[0], 1);
    __threadfence();
}

//consumer
__global__
void app_one_per_warp_test(int* size_to_alloc,
                           int* iter_num,
                           int MONO,
                           Runtime runtime){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int* ptr_tab[32];
    int warp_id = threadIdx.x/32;
    
    for (int i=0; i<iter_num[0]; ++i){
        // ALLOCTAION
        volatile int* new_ptr = NULL;
        runtime.malloc_warp(&new_ptr, &ptr_tab[warp_id], size_to_alloc[0]);
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

//consumer
__global__
void app_async_one_per_warp_test(int* size_to_alloc,
                                 int* iter_num,
                                 int MONO,
                                 Runtime runtime){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int* ptr_tab[32];

    int warp_id = threadIdx.x/32;
    int lane_id = threadIdx.x%32;
    if (lane_id == 0) ptr_tab[warp_id] = NULL;

    __syncthreads();

    for (int i=0; i<iter_num[0]; ++i){
        __threadfence();
        volatile int* new_ptr = NULL;
        //Future future_ptr;
        //runtime.malloc_warp(future_ptr, size_to_alloc[0]);

        if (lane_id == 0){
            runtime.malloc_async((volatile int**)&ptr_tab[warp_id], 32*size_to_alloc[0]);
            //runtime.malloc_async(future_ptr, 32*size_to_alloc[0]);
            /** copied from request **/
            // wait for request to be completed
            while (runtime.is_working()){
                if (runtime.is_finished(thid)){
                    runtime.request_processed((request_type)MALLOC, (volatile int**)&ptr_tab[warp_id]);
                    break;
                }
                __threadfence();
            }
            /* copied from request - end**/
            //assert(ptr_tab[warp_id]);
        }
        __threadfence();
        __syncthreads();
        int offset = lane_id * size_to_alloc[0];
        new_ptr = (volatile int*)(((volatile char*)(ptr_tab[warp_id])) + offset);
        __syncthreads();
        __threadfence();
        new_ptr[0] = thid;
        assert(new_ptr[0] == thid);
        __threadfence();
        __syncthreads();
        if (lane_id == 0){
            runtime.free_async((volatile int**)&ptr_tab[warp_id]);
            /** copied from request **/
            // wait for request to be completed
            while (runtime.is_working()){
                if (runtime.is_finished(thid)){
                    runtime.request_processed((request_type)FREE, (volatile int**)&ptr_tab[warp_id]);
                    break;
                }
                __threadfence();
            }
            /** copied from request - end**/
        }
        __threadfence();
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}

//consumer
__global__
void app_async_request_test(int* size_to_alloc,
                            int* iter_num,
                            int MONO,
                            Runtime runtime){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i=0; i<iter_num[0]; ++i){

        // MEMORY ALLOCATION
        __threadfence();
        Future future_ptr; //Can I pass it outside of the kernel?
        runtime.malloc_async(future_ptr, size_to_alloc[0]);
        
        // WRITE TEST
        volatile int* new_ptr = future_ptr.get();
        new_ptr[0] = thid;
        __threadfence();

        // READ TEST
        assert(new_ptr[0] == thid);

        // MEMORY RECLAMATION
        __threadfence();
        runtime.free_async(future_ptr);;
        __threadfence();
        runtime.wait((request_type)FREE, thid, &future_ptr.ptr);
    }
    atomicAdd((int*)(runtime.exit_counter), 1);
    __threadfence();
}

//consumer
__global__
void app_test(int* size_to_alloc,
              int* iter_num,
              int MONO, 
              Runtime runtime){
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

void check_persistent_kernel_results(int* exit_signal, 
                   int* exit_counter, 
                   int block_size, 
                   int app_grid_size, 
                   RequestType& requests, 
                   int requests_num,
                   bool& finish){

    // Check results
    int old_counter = -1;
    long long int iter = 0;
    long long int time_limit = 1000000000;
    //printf("waiting till allocations are done\n");
    while (iter < time_limit){
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        //if (iter%60 == 0)
        //    printf("%lld min, exit counter %d\n", iter/60, exit_counter[0]);
        // Check if all allocations are done
        if (exit_counter[0] == block_size*app_grid_size){
            GUARD_CU(cudaStreamSynchronize(0));
            GUARD_CU(cudaPeekAtLastError());
            finish = true;
            break;
        }else{
            GUARD_CU(cudaPeekAtLastError());
            if (exit_counter[0] != old_counter){
                old_counter = exit_counter[0];
                //printf("%d\n", old_counter);
                iter = 0;
            }
            ++iter;
        }
        if (iter >= time_limit){
            // Start mm and app again
            printf("time limit exceed, break\n");
            fflush(stdout);
            *exit_signal = 1;
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
        }
    }
    GUARD_CU(cudaPeekAtLastError());
}

void createStreams(cudaStream_t& gc_stream, 
                   cudaStream_t& mm_stream, 
                   cudaStream_t& app_stream){
    GUARD_CU(cudaStreamCreateWithFlags( &gc_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaStreamCreateWithFlags( &mm_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
}

void start_memory_manager(PerfMeasure& timing_mm, 
                          uint32_t mm_grid_size,
                          uint32_t block_size, 
                          CUcontext& mm_ctx,
                          Runtime& runtime){
    timing_mm.startMeasurement();
    assert(runtime.exit_signal);
  
    void *args[] = {&runtime};
    printf("go mem manager kernel\n");
    fflush(stdout);
    GUARD_CU(cudaLaunchCooperativeKernel((void*)mem_manager, mm_grid_size, block_size, args));
    //GUARD_CU(cudaLaunchKernel((void*)mem_manager, mm_grid_size, block_size, args));
    printf("mem manager done\n");
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
    assert(runtime.exit_signal);
    //assert(runtime->gc);
    printf("hello gc\n");
    fflush(stdout);
    
    void *args[] = {&runtime};
    printf("go gc kernel\n");
    fflush(stdout);
    GUARD_CU(cudaLaunchCooperativeKernel((void*)garbage_collector, gc_grid_size, block_size, args));
    //GUARD_CU(cudaLaunchKernel((void*)garbage_collector, gc_grid_size, block_size, args));
    printf("gc done\n");
    GUARD_CU((cudaError_t)cudaGetLastError());
    GUARD_CU(cudaPeekAtLastError());
   
    timing_gc.stopMeasurement();
}

void start_callback_server(PerfMeasure& timing_cb, 
                          uint32_t cb_grid_size,
                          uint32_t block_size, 
                          CUcontext& cb_ctx,
                          Runtime& runtime){
    timing_cb.startMeasurement();
    while (! runtime.exit_signal[0]){
        // TODO ??? 
        std::this_thread::sleep_for(std::chrono::microseconds(1));
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
    runtime.requests->free();
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void start_application(int type, 
                       PerfMeasure& timing_sync, 
                       uint32_t grid_size,
                       uint32_t block_size, 
                       CUcontext& ctx,
                       volatile int* exit_signal,
                       RequestType* requests,
                       volatile int* exit_counter,
                       int* dev_size_to_alloc, 
                       int* dev_iter_num,
                       int mono, 
                       bool& kernel_complete,
                       MemoryManagerType& memory_manager, 
                       Runtime& runtime){
    //printf("requests %x\n", requests);
    //printf("d_memory %x\n", requests->d_memory);
    //assert(requests->d_memory);
    auto dev_mm = memory_manager.getDeviceMemoryManager();
    if (mono == MPS_mono){
        void* args[] = {&(requests->d_memory), &exit_counter, &dev_size_to_alloc, &dev_iter_num, &dev_mm};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        debug("start application: MPS mono!\n");
        //GUARD_CU(cudaLaunchKernel((void*)mono_app_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)mono_app_test, grid_size, block_size, args, 0, 0));
        debug("done!\n");
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        debug("ctx sync done!\n");
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();

        fflush(stdout);
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == one_per_warp){
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        debug("start application one per warp\n");
        //GUARD_CU(cudaLaunchKernel((void*)app_one_per_warp_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_one_per_warp_test, grid_size, block_size, args));
        debug("stop application one per warp\n");
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        debug("stop application one per warp\n");
        GUARD_CU(cudaPeekAtLastError());
        debug("stop application one per warp\n");
        timing_sync.stopMeasurement();
        fflush(stdout);
        debug("stop application one per warp\n");
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == async_request){
        debug("start applications: type %d\n", type);
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        //GUARD_CU(cudaLaunchKernel((void*)app_async_request_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_async_request_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }else if (mono == async_one_per_warp){
        debug("start applications: type %d\n", type);
        void* args[] = {&dev_size_to_alloc, &dev_iter_num, &mono, &runtime};
        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        //GUARD_CU(cudaLaunchKernel((void*)app_async_request_test, grid_size, block_size, args, 0, 0));
        GUARD_CU(cudaLaunchCooperativeKernel((void*)app_async_one_per_warp_test, grid_size, block_size, args));
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        //GUARD_CU(cudaProfilerStop());
    }else{
        debug("start applications: type %d\n", type);
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

void sync_streams(cudaStream_t& gc_stream, 
                  cudaStream_t& mm_stream, 
                  cudaStream_t& app_stream){

    debug("waiting for streams\n");
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
    debug("app stream synced\n");
    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaPeekAtLastError());
    debug("mm stream synced\n");
    GUARD_CU(cudaStreamSynchronize(gc_stream));
    GUARD_CU(cudaPeekAtLastError());
    debug("gc stream synced\n");
    GUARD_CU(cudaPeekAtLastError());

}

void simple_monolithic_app(int mono, int kernel_iteration_num, 
            int size_to_alloc, size_t* ins_size, size_t num_iterations, 
            int SMs, int* sm_app, int* sm_mm, int* sm_gc, int* allocs, 
            float* uni_req_per_sec, int* array_size, int block_size, 
            int device_id){

    auto instant_size = *ins_size;
    CUcontext default_ctx;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));

    //Ouroboros initialization
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    
    volatile int* exit_signal;
    volatile int* exit_counter;
    int* dev_size_to_alloc;
    int* dev_kernel_iteration_num;
    
    allocManaged(&exit_signal, sizeof(int32_t));
    allocManaged(&exit_counter, sizeof(uint32_t));
    GUARD_CU(cudaMallocManaged(&dev_size_to_alloc, sizeof(int)));
    GUARD_CU(cudaMallocManaged(&dev_kernel_iteration_num, sizeof(int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
   
    *dev_size_to_alloc = size_to_alloc;
    *dev_kernel_iteration_num = kernel_iteration_num;
    
    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, device_id));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_size_to_alloc, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_kernel_iteration_num, sizeof(int), device, NULL));

    //int block_size = 1024;
    //int block_size = 256;
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

        auto dev_mm = memory_manager.getDeviceMemoryManager();
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

void mps_monolithic_app(int mono, int kernel_iteration_num, int size_to_alloc, 
            size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
            int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
            int* array_size, int block_size, int device_id){

    auto instant_size = *ins_size;

    //Ouroboros initialization
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    
    volatile int* exit_signal;
    allocManaged(&exit_signal, sizeof(int32_t));

    volatile int* exit_counter;
    allocManaged(&exit_counter, sizeof(uint32_t));

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
    //int block_size = 1024;
    //int block_size = 256;
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
        runtime.init(requests_num, device, memory_manager);

        printf("request addr %x\n", runtime.requests);

        // Run APP (all threads do malloc)
        bool kernel_complete = false;
        std::thread app_thread{[&] {
            GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
            //GUARD_CU((cudaError_t)cuCtxSynchronize());
            debug("start app\n");
            //malloc_total_sync.startMeasurement();

            //mps_monolithic_app
            start_application(MALLOC, malloc_total_sync, 
                    app_grid_size, block_size, app_ctx, exit_signal,
                    runtime.requests, exit_counter, dev_size_to_alloc, 
                    dev_kernel_iteration_num, mono, kernel_complete, 
                    memory_manager, runtime);
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

void mps_app(int mono, int kernel_iteration_num, int size_to_alloc, 
        size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
        int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
        int* array_size, int block_size, int device_id){

    auto instant_size = *ins_size;
    CUcontext default_ctx;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));

    //Ouroboros initialization
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);

    int total_gpus = 0;
    GUARD_CU((cudaError_t)cudaGetDeviceCount(&total_gpus));
    printf("device %d/%d\n", device_id, total_gpus);
    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, device_id));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    //Create two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t gc_stream, mm_stream, app_stream;
    createStreams(gc_stream, mm_stream, app_stream);

    int* dev_size_to_alloc;        
    int* dev_kernel_iteration_num; 
    allocManaged_(&dev_size_to_alloc, sizeof(int));
    allocManaged_(&dev_kernel_iteration_num, sizeof(int));
    *dev_size_to_alloc = size_to_alloc;
    *dev_kernel_iteration_num = kernel_iteration_num;
    
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_size_to_alloc, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync(
            (int*)dev_kernel_iteration_num, sizeof(int), device, NULL));

    int it = 0;
    printf("Block size = %d\n", block_size);
    //int block_size = 1024;
    //int block_size = 256;
    //SMs -= 10;

    int cb_grid_size = 1;

    for (int app_grid_size=55; app_grid_size<SMs-cb_grid_size; ++app_grid_size){
    //for (int app_grid_size=1; app_grid_size<2; ++app_grid_size){
        for (int mm_grid_size=1; mm_grid_size<(SMs-app_grid_size-cb_grid_size); ++mm_grid_size){
        //for (int mm_grid_size=1; mm_grid_size<=app_grid_size; ++mm_grid_size){

            //for (int gc_grid_size=1; gc_grid_size<=app_grid_size; ++gc_grid_size){

                //if (app_grid_size + mm_grid_size + gc_grid_size > SMs) 
                   // continue;

                int gc_grid_size = SMs - app_grid_size - mm_grid_size - cb_grid_size;
                ////int gc_grid_size = 1;
                if (gc_grid_size < 1) continue;
                //if (gc_grid_size > app_grid_size) continue;

                int requests_num{app_grid_size * block_size};

                debug("SMs: app %d, mm %d, gc %d, total %d\n", 
                        app_grid_size, mm_grid_size, gc_grid_size, SMs);
                debug("requests_num %d\n", requests_num);
                //fflush(stdout);

                //output
                sm_app[it] = app_grid_size;
                sm_mm [it] = mm_grid_size;
                sm_gc [it] = gc_grid_size;
                allocs[it] = requests_num;

                //int mul = 1;
                int app_numBlocksPerSm = 1;// 0;
                int gc_numBlocksPerSm =  1;//0;
                int mm_numBlocksPerSm =  1;//0;
                int cb_numBlocksPerSm =  1;
                
                debug("num blocks per sm by cudaOccMaxActBlPerSM: app %d, mm %d, gc %d\n", app_numBlocksPerSm, gc_numBlocksPerSm, mm_numBlocksPerSm);
                //fflush(stdout);

                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU((cudaError_t)cudaGetLastError());

                CUexecAffinityParam_v1 app_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                    (unsigned int)app_grid_size * app_numBlocksPerSm};
                CUexecAffinityParam_v1 mm_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                    (unsigned int)mm_grid_size  *  mm_numBlocksPerSm};
                CUexecAffinityParam_v1 gc_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                    (unsigned int)gc_grid_size  *  gc_numBlocksPerSm};
                CUexecAffinityParam_v1 cb_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                    (unsigned int)cb_grid_size  *  cb_numBlocksPerSm};

                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU((cudaError_t)cudaGetLastError());

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

                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU((cudaError_t)cudaGetLastError());

                GUARD_CU((cudaError_t)cuCtxCreate_v3(
                            &mm_ctx, &mm_param, 1, affinity_flags, device));

                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU((cudaError_t)cudaGetLastError());

                GUARD_CU((cudaError_t)cuCtxCreate_v3(
                            &gc_ctx, &gc_param, 1, affinity_flags, device));

                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU((cudaError_t)cudaGetLastError());

                GUARD_CU((cudaError_t)cuCtxCreate_v3(
                            &cb_ctx, &cb_param, 1, affinity_flags, device));

                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU((cudaError_t)cudaGetLastError());

                //Timing variables
                PerfMeasure malloc_total_sync, timing_mm, timing_gc, timing_cb;
                for (int iteration = 0; iteration < num_iterations; ++iteration){
                    Runtime runtime;
                    runtime.init(requests_num, device, memory_manager);

                    printf("iteration %d\n", iteration);

                    GUARD_CU(cudaDeviceSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    GUARD_CU((cudaError_t)cudaGetLastError());
    
                    fflush(stdout);
                    debug("start threads\n");
                    // Run Memory Manager (Presistent kernel)
                    std::thread mm_thread{[&] {
                        GUARD_CU((cudaError_t)cuCtxSetCurrent(mm_ctx));
                        GUARD_CU((cudaError_t)cuCtxSynchronize());
                        GUARD_CU(cudaPeekAtLastError());
                        debug("start mm\n");
                        start_memory_manager(timing_mm, 
                                mm_numBlocksPerSm*mm_grid_size, 
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
                        start_garbage_collector(timing_gc, 
                                gc_numBlocksPerSm*gc_grid_size, 
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
                        start_callback_server(timing_cb, 
                                cb_numBlocksPerSm*cb_grid_size, 
                                block_size, cb_ctx, runtime);
                        debug("callback done, sync\n");
                        GUARD_CU((cudaError_t)cuCtxSynchronize());
                        GUARD_CU(cudaPeekAtLastError());
                        debug("done\n");
                    }};
                    //std::this_thread::sleep_for(std::chrono::seconds(1));

                    //printf("-");
                    fflush(stdout);
                    //while (!(*gc_started && *mm_started && *cb_started));
                    while (!(runtime.gc->is_running() && 
                             runtime.mm->is_running() //&& 
                             //runtime.cb.is_running()
                             ));
                    GUARD_CU((cudaError_t)cudaGetLastError());

                    debug("app_numBlocksPerSm %d, app_grid_size %d, block_size %d\n", app_numBlocksPerSm, app_grid_size, block_size);
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
                        start_application(MALLOC, malloc_total_sync, 
                                app_numBlocksPerSm*app_grid_size, 
                                block_size, app_ctx, runtime.exit_signal, 
                                runtime.requests, runtime.exit_counter, 
                                dev_size_to_alloc, 
                                dev_kernel_iteration_num, mono, 
                                kernel_complete, memory_manager, 
                                runtime);
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
                    case async_request:
                        printf("Async request. "); 
                        break;
                    case async_one_per_warp:
                        printf("Async one per warp. ");
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
            //}
        }
    }
    *array_size = it;
    }

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


// TODO add the device id to run tests concurrently
void pmm_init(int mono, int kernel_iteration_num, int size_to_alloc, 
        size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
        int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
        int* array_size, int block_size, int device){

    GUARD_CU(cudaSetDevice(device));

//    printf("size to alloc per thread %d, num iterations %lu, kernel iterations %d, instantsize %lu, mono %d\n", 
//            size_to_alloc, num_iterations, kernel_iteration_num, 
//            *ins_size, mono);

    //std::cout << "#requests\t" << "#sm app\t\t" << "#sm mm\t\t" << 
    //            "#sm gc\t\t" << "#malloc and free per sec\n";

    printf("mono %d\n", mono);
    printf("device %d\n", device);

    if (mono == MPS_mono){
        printf("MPS_mono\n");

        mps_monolithic_app(mono, kernel_iteration_num, 
                size_to_alloc, ins_size, num_iterations, 
                SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device);

        printf("MPS_mono\n");
    }else if (mono == simple_mono){
        printf("simple mono\n");

        simple_monolithic_app(mono, kernel_iteration_num, 
                size_to_alloc, ins_size, num_iterations, 
                SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device);

        printf("simple mono\n");
    }else if (mono == one_per_warp){
        printf("one per warp\n");

        mps_app(mono, kernel_iteration_num, size_to_alloc, ins_size, 
                num_iterations, SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device);

        printf("one per warp\n");
    }else if (mono == async_request){
        printf("async request\n");

        mps_app(mono, kernel_iteration_num, size_to_alloc, ins_size, 
                num_iterations, SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device);

        printf("async request\n");
    }else if (mono == async_one_per_warp){
        printf("async one per warp\n");
        
        mps_app(mono, kernel_iteration_num, size_to_alloc, ins_size, 
                num_iterations, SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device);

        printf("async one per warp\n");
    }else{
        printf("MPS services\n");

        mps_app(mono, kernel_iteration_num, size_to_alloc, ins_size, 
                num_iterations, SMs, sm_app, sm_mm, sm_gc, allocs, 
                uni_req_per_sec, array_size, block_size, device);

        printf("MPS services\n");
    }
}

}

