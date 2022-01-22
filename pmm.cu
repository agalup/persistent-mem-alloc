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

//#include "src/gpu_hash_table.cuh"

using namespace std;

extern "C" {

__global__
    void mem_free(volatile int** d_memory, 
            volatile int* request_id, 
            MemoryManagerType* mm, 
            volatile int* requests_num){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= requests_num[0]){
        return;
    }
    __threadfence();
    if (d_memory[thid]){
        printf("sync error: %d was not released before\n", thid);
        mm->free((void*)d_memory[thid]);
    }
}

__device__
void _request_processing(
        int request_id, 
        volatile int* exit_signal,
        volatile int* request_signal,
        volatile int* request_counter,
        volatile int* request_ids, 
        volatile int** request_dest, 
        MemoryManagerType* mm, 
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock){

    // SEMAPHORE
    acquire_semaphore((int*)lock, request_id);
    debug("mm: request recieved %d\n", request_id); 
    auto addr_id = request_ids[request_id];
    
    switch (request_signal[request_id]){

        case MALLOC:
            if (addr_id == -1){
                addr_id = atomicAdd((int*)&request_counter[0], 1);
                request_ids[request_id] = addr_id;
            }else{
                assert(d_memory[addr_id] == NULL);
            }
            __threadfence();
            d_memory[addr_id] = reinterpret_cast<volatile int*>
                (mm->malloc(4+request_mem_size[request_id]));
            __threadfence();
            assert(d_memory[addr_id]);
            d_memory[addr_id][0] = 0;
            request_dest[request_id] = &d_memory[addr_id][1];
            atomicExch((int*)&request_signal[request_id], request_done);
            break;

        case FREE:
            assert(d_memory[addr_id]);
            if (d_memory[addr_id][0] != 0)
                printf("d_memory{%d} = %d\n", addr_id, d_memory[addr_id][0]);
            assert(d_memory[addr_id][0] == 0);
            auto request_status = d_memory[addr_id][0] - 1;
            d_memory[addr_id][0] -= 1;
            request_dest[request_id] = NULL;
            assert(d_memory[addr_id][0] == -1);
            if (request_status < 0){
                atomicExch((int*)&request_signal[request_id], request_gc);
            }else{
                assert(1);
                printf("should not be here!\n");
                atomicExch((int*)&request_signal[request_id], request_done);
            }
            break;

        case GC:
            assert(d_memory[addr_id]);
            assert(d_memory[addr_id][0] == -1);
            __threadfence();
            mm->free((void*)d_memory[addr_id]);
            __threadfence();
            d_memory[addr_id] = NULL;
            atomicExch((int*)&request_signal[request_id], request_done);
            break;

        default:
            printf("request processing fail\n");

    }
    release_semaphore((int*)lock, request_id);
    // SEMAPHORE
}

__global__
void garbage_collector(volatile int** d_memory,
                       volatile int* requests_number, 
                       volatile int* request_counter,
                       volatile int* request_signal, 
                       volatile int* request_ids, 
                       volatile int* request_mem_size,
                       volatile int* lock,
                       volatile int* exit_signal,
                       MemoryManagerType* mm){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (! exit_signal[0]){
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            __threadfence();
            if (request_signal[request_id] == GC){
                _request_processing(request_id, exit_signal, request_signal,
                                    request_counter, request_ids, NULL, mm, 
                                    d_memory, request_mem_size, lock);
                __threadfence();
            }
        }
        __threadfence();
    }
}


//producer
__global__
void mem_manager(volatile int* exit_signal, 
        volatile int* requests_number, 
        volatile int* request_counter,
        volatile int* request_signal, 
        volatile int* request_ids, 
        volatile int** request_dest,
        MemoryManagerType* mm, 
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (! exit_signal[0]){
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            __threadfence();
            if (request_signal[request_id] == MALLOC or 
                request_signal[request_id] == FREE){
                _request_processing(request_id, exit_signal, request_signal, 
                                    request_counter, request_ids, request_dest,
                                    mm, d_memory, request_mem_size, lock);
                __threadfence();
            
            debug("mm: request done %d\n", request_id);
            }
        }
        __threadfence();
    }
}

__device__
void post_request(request_type type,
                  volatile int** dest,
                  volatile int* lock,
                  volatile int* request_mem_size,
                  volatile int* request_id,
                  volatile int* request_signal,
                  volatile int** request_dest,
                  volatile int* exit_signal,
                  int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    __threadfence();
    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    if (type == MALLOC){
        request_mem_size[thid] = size_to_alloc;
    }
    // SIGNAL update
    atomicExch((int*)&request_signal[thid], type);
    release_semaphore((int*)lock, thid);
    __threadfence();
    // SEMAPHORE
}

__device__
void request_processed(request_type type,
                      volatile int* lock,
                      volatile int* request_id,
                      volatile int* exit_signal,
                      volatile int** d_memory,
                      volatile int** dest,
                      volatile int* request_signal,
                      volatile int** request_dest,
                      int& req_id){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // SEMAPHORE
    __threadfence();
    acquire_semaphore((int*)lock, thid);
    switch (type){
        case MALLOC:
            req_id = request_id[thid];
            if (req_id >= 0 && !exit_signal[0]) {
                *dest = request_dest[thid];
                assert(d_memory[req_id] != NULL);
                if (d_memory[req_id][0] != 0)
                    printf("d_memory[%d] = %d\n", req_id, d_memory[req_id][0]);
                //assert(d_memory[req_id][0] == 0);
                assert(*dest != NULL);
                assert(request_dest[thid] == *dest);
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
    request_signal[thid] = request_empty;
    release_semaphore((int*)lock, thid);
    __threadfence();
    // SEMAPHORE
}

__device__
void request(request_type type,
        volatile int* exit_signal,
        volatile int** d_memory,
        volatile int** dest,
        volatile int* request_signal,
        volatile int* request_mem_size, 
        volatile int* request_id,
        volatile int** request_dest,
        volatile int* lock,
        int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int req_id = -1;
    // wait for success
    while (!exit_signal[0]){
        if (request_signal[thid] == request_empty){
            post_request(type, dest, lock, request_mem_size, request_id, 
                        request_signal, request_dest, exit_signal, size_to_alloc);
            break;
        }
        __threadfence();
    }

    __threadfence();

    int it = 0;
    // wait for success
    while (!exit_signal[0]){
        char* type_ = new char[10];
        char* state = new char[10];
        if (++it > 1000){
            if (type == MALLOC) type_ = "MALLOC"; else type_ = "FREE";
            switch (request_signal[thid]){
                case request_empty:  state = "EMPTY";  break;
                case request_done:   state = "DONE";   break;
                case request_malloc: state = "MALLOC"; break;
                case request_free:   state = "FREE";   break;
                case request_gc:     state = "GC";     break;
            }
            printf("thid %d, current state %s\n", thid, state);
        }

        if (request_signal[thid] == request_done){
            request_processed(type, lock, request_id, exit_signal, d_memory, 
                        dest, request_signal, request_dest, req_id);
            break;
        }
        __threadfence();
    }
}

//consumer
__global__
void malloc_app_test(volatile int* exit_signal,
        volatile int** d_memory, 
        volatile int* request_signal, 
        volatile int* request_mem_size,
        volatile int* request_id, 
        volatile int** request_dest, 
        volatile int* exit_counter, 
        volatile int* lock,
        int size_to_alloc,
        int iter_num){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    //if (thid == 0)
    //    printf("start inner iteration\n");
    for (int i=0; i<iter_num; ++i){
        //if (thid == 0) printf("inner iteration %d\n", i);
        __threadfence();

        volatile int* new_ptr = NULL;
        request((request_type)MALLOC, exit_signal, d_memory, &new_ptr, 
                request_signal, request_mem_size, request_id, request_dest,
                lock, size_to_alloc);
        new_ptr[0] = thid;

        __threadfence();

        assert(d_memory[request_id[thid]]);
        int value = d_memory[request_id[thid]][0];
        if (value != 0) printf("val = %d\n", value);
        assert(new_ptr[0] == thid);
        
        __threadfence();

        request((request_type)FREE, exit_signal, d_memory, &new_ptr,
                request_signal, request_mem_size, request_id, request_dest,
                lock, size_to_alloc);
        
        __threadfence_system();
    }
    //if (thid == 0)
    //    printf("inner iteration done\n");
    
    atomicAdd((int*)&exit_counter[0], 1);

    __threadfence();

    //printf("exit counter[%d] = %d\n", thid, exit_counter[0]);
}

//consumer2
__global__
void free_app_test(volatile int* exit_signal, 
              volatile int** d_memory, 
              volatile int* request_signal, 
              volatile int* request_mem_size,
              volatile int* request_id, 
              volatile int** request_dest, 
              volatile int* exit_counter, 
              volatile int* lock,
              int size_to_alloc,
              int iter_num){
    
    __threadfence();
   
    request((request_type)FREE, exit_signal, d_memory, NULL, 
            request_signal, request_mem_size, request_id, request_dest,
            lock, 0);

    atomicAdd((int*)&exit_counter[0], 1);
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
        //std::this_thread::sleep_for(std::chrono::seconds(1));
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

void allocManaged(int** ptr, size_t size){
    GUARD_CU(cudaMallocManaged(ptr, size));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
}

void start_memory_manager(PerfMeasure& timing_mm, 
                          uint32_t mm_grid_size,
                          uint32_t block_size, 
                          CUcontext& mm_ctx,
                          int* exit_signal,
                          RequestType& requests,
                          MemoryManagerType& memory_manager){

    timing_mm.startMeasurement();
    mem_manager<<<mm_grid_size, block_size>>>(exit_signal, 
            requests.requests_number, requests.request_counter, requests.request_signal, 
            requests.request_id, requests.request_dest,
#ifdef OUROBOROS__
            memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
            memory_manager,
#endif
#endif
            requests.d_memory, requests.request_mem_size, requests.lock);
    timing_mm.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());

}

void start_garbage_collector(PerfMeasure& timing_gc, 
                          uint32_t gc_grid_size,
                          uint32_t block_size, 
                          CUcontext& gc_ctx,
                          int* exit_signal,
                          RequestType& requests,
                          MemoryManagerType& memory_manager){
    timing_gc.startMeasurement();
    garbage_collector<<<gc_grid_size, block_size>>>(
            requests.d_memory, 
            requests.requests_number,
            requests.request_counter,
            requests.request_signal,
            requests.request_id,
            requests.request_mem_size, 
            requests.lock,
            exit_signal,
#ifdef OUROBOROS__
            memory_manager.getDeviceMemoryManager()
#else
#ifdef HALLOC__
            memory_manager
#endif
#endif
            );
    timing_gc.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());

}

void clean_memory(uint32_t grid_size,
                  uint32_t block_size, 
                  RequestType& requests,
                  MemoryManagerType& memory_manager,
                  int* exit_signal){

    *exit_signal = 1;
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    mem_free<<<grid_size, block_size>>>(requests.d_memory, 
            requests.request_id, 
#ifdef OUROBOROS__
            memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
            memory_manager,
#endif
#endif
            requests.requests_number);

    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
    requests.free();
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}


void start_application(int type, 
                       PerfMeasure& timing_launch, 
                       PerfMeasure& timing_sync, 
                       uint32_t grid_size,
                       uint32_t block_size, 
                       CUcontext& ctx,
                       int* exit_signal,
                       RequestType& requests,
                       int* exit_counter,
                       int size_to_alloc, 
                       int iter_num,
                       bool& kernel_complete){
    // Run application
    //timing_launch.startMeasurement();
    timing_sync.startMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    auto kernel = malloc_app_test;
    if (type == FREE){
        kernel = free_app_test;
    }
    //printf("start kernel\n");
    kernel<<<grid_size, block_size>>>(exit_signal, requests.d_memory, 
            requests.request_signal, requests.request_mem_size, 
            requests.request_id, requests.request_dest, exit_counter, requests.lock, 
            size_to_alloc, iter_num);
    //printf("kernel done, exit counter %d\n", exit_counter[0]);
    GUARD_CU(cudaPeekAtLastError());
    //timing_launch.stopMeasurement();

    // Check resutls: test
    //printf("check results\n");
    //fflush(stdout);
    check_persistent_kernel_results(exit_signal, exit_counter, block_size, 
            grid_size, requests, requests.size, kernel_complete);
    timing_sync.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    //printf("results done\n");
    //fflush(stdout);

    if (kernel_complete){
        if (type == MALLOC){
            /*printf("test1!\n");
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            test1<<<grid_size, block_size, 0, stream>>>(requests.d_memory, 
                                                        requests.size);
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            mem_test((int**)requests.d_memory, requests.size, grid_size, block_size);
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            printf("test done\n");*/
        }else if (type == FREE){
            debug("test2!\n");
            GUARD_CU(cudaStreamSynchronize(0));
            GUARD_CU(cudaPeekAtLastError());
            test2<<<grid_size, block_size>>>(requests.d_memory, 
                                                        requests.size);
            GUARD_CU(cudaStreamSynchronize(0));
            GUARD_CU(cudaPeekAtLastError());
        }
    }
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


void pmm_init(int kernel_iteration_num, int size_to_alloc, size_t* ins_size, 
              size_t num_iterations, int SMs, int* sm_app, int* sm_mm, int* sm_gc, 
              int* allocs, float* uni_req_per_sec, int* array_size){

    auto instant_size = *ins_size;

    CUcontext default_ctx;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));

#ifdef OUROBOROS__
    //Ouroboros initialization
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);
#else
#ifdef HALLOC__
    //Halloc initialization
    MemoryManagerType memory_manager(instant_size);
#endif
#endif
 
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t gc_stream, mm_stream, app_stream;
    createStreams(gc_stream, mm_stream, app_stream);
    
    int* exit_signal;
    allocManaged(&exit_signal, sizeof(int32_t));

    int* exit_counter;
    allocManaged(&exit_counter, sizeof(uint32_t));

    int block_size = 1024;
    printf("size to alloc per thread %d, num iterations %d, kernel iterations %d, instantsize %ld\n", 
                size_to_alloc, num_iterations, kernel_iteration_num, instant_size);
    std::cout << "#requests\t" << "#sm app\t\t" << "#sm mm\t\t" << "#sm gc\t\t" << "#malloc and free per sec\n";

    int it = 0;

    for (int app_grid_size=1; app_grid_size<SMs; ++app_grid_size){
    //for (int app_grid_size = 1; app_grid_size < 5; ++app_grid_size){

    for (int mm_grid_size=1; mm_grid_size<(SMs-app_grid_size); ++mm_grid_size){
    //for (int mm_grid_size = 1; mm_grid_size < 5; ++mm_grid_size){

        int gc_grid_size = SMs - app_grid_size - mm_grid_size;
        if (gc_grid_size <= 0) continue;

        int requests_num{app_grid_size*block_size};

        printf("SMs: app %d, mm %d, gc %d, total %d\n", app_grid_size, mm_grid_size, gc_grid_size, SMs);
        printf("requests_num %d\n", requests_num);

        //output
        sm_app[it] = app_grid_size;
        sm_mm [it] = mm_grid_size;
        sm_gc [it] = gc_grid_size;
        allocs[it] = requests_num;

        CUexecAffinityParam_v1 app_param{
        CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)app_grid_size};
        CUexecAffinityParam_v1 mm_param{
        CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)mm_grid_size};
        CUexecAffinityParam_v1 gc_param{
        CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)gc_grid_size};

        auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
        CUcontext app_ctx, mm_ctx, gc_ctx;
        CUdevice device;
        GUARD_CU((cudaError_t)cuDeviceGet(&device, 0));

        GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx, &app_param, 1, affinity_flags, device));
        GUARD_CU((cudaError_t)cuCtxCreate_v3(&mm_ctx, &mm_param, 1, affinity_flags, device));
        GUARD_CU((cudaError_t)cuCtxCreate_v3(&gc_ctx, &gc_param, 1, affinity_flags, device));
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());

        //Timing variables
        PerfMeasure timing_malloc_app, timing_mm, timing_gc, malloc_total_sync;

        for (int iteration = 0; iteration < num_iterations; ++iteration){
            printf("iteartion %d\n", iteration);

            *exit_signal = 0;
            *exit_counter = 0;
            RequestType requests;
            requests.init(requests_num);
            requests.memset();

            //GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));
            printf("start threads\n");

            // Run Memory Manager (Presistent kernel)
            std::thread mm_thread{[&] {
                GUARD_CU((cudaError_t)cuCtxSetCurrent(mm_ctx));
                //GUARD_CU((cudaError_t)cuCtxSynchronize());
                printf("start mm\n");
                start_memory_manager(timing_mm, mm_grid_size, block_size, mm_ctx,
                                 exit_signal, requests, memory_manager);
                printf("mm done, sync\n");
                GUARD_CU((cudaError_t)cuCtxSynchronize());
                GUARD_CU(cudaPeekAtLastError());
                printf("done\n");
            }};

            //std::this_thread::sleep_for(std::chrono::seconds(1));

            // Run Garbage Collector (persistent kernel)
            std::thread gc_thread{[&] {
                GUARD_CU((cudaError_t)cuCtxSetCurrent(gc_ctx));
                //GUARD_CU((cudaError_t)cuCtxSynchronize());
                printf("start gc\n");
                start_garbage_collector(timing_gc, gc_grid_size, block_size, gc_ctx,
                                 exit_signal, requests, memory_manager);
                printf("gc done, sync\n");
                GUARD_CU((cudaError_t)cuCtxSynchronize());
                GUARD_CU(cudaPeekAtLastError());
                printf("done\n");
            }}; 
        
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // Run APP (all threads do malloc)
            bool kernel_complete = false;
            std::thread app_thread{[&] {
                GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
                //GUARD_CU((cudaError_t)cuCtxSynchronize());
                printf("start app\n");
                start_application(MALLOC, timing_malloc_app, malloc_total_sync, 
                              app_grid_size, block_size, app_ctx, exit_signal,
                              requests, exit_counter, size_to_alloc, 
                              kernel_iteration_num, kernel_complete);
                printf("app done, sync\n");
                GUARD_CU((cudaError_t)cuCtxSynchronize());
                GUARD_CU(cudaPeekAtLastError());
                printf("done\n");
            }};

            //std::this_thread::sleep_for(std::chrono::seconds(1));

            printf("join app\n");
            app_thread.join();
            printf("app joined\n");

            if (not kernel_complete){
                printf("kernel is not completed, free memory which app allocated\n");
                clean_memory(app_grid_size, block_size, requests, memory_manager, exit_signal);
                continue;
            }

            *exit_signal = 1;

            printf("join mm\n");
            mm_thread.join();
            printf("mm joined\n");
      
            printf("join gc\n");
            gc_thread.join();
            printf("gc joined\n");
           
            // Deallocate device memory
            //cuCtxSetCurrent(default_ctx);
            //GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
            clean_memory(app_grid_size, block_size, requests, memory_manager, exit_signal);

            //GUARD_CU((cudaError_t)cuCtxSynchronize());
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
        }
        printf("done\n");

        GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));
        GUARD_CU((cudaError_t)cuCtxDestroy(gc_ctx));
        GUARD_CU((cudaError_t)cuCtxDestroy(mm_ctx));
        GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        
        // Output: the number of requests done per a second
        auto malloc_total_sync_time   = malloc_total_sync.generateResult();
        uni_req_per_sec[it]   = (requests_num * 1000.0)/(malloc_total_sync_time.mean_/kernel_iteration_num);

        printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
            app_grid_size, mm_grid_size, gc_grid_size, uni_req_per_sec[it]);

        ++it;
    }
    }
    
    *array_size = it;
}

}
