#include <iostream>
#include <string>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>
#include <any>

#include "PerformanceMeasure.cuh"
#include "cuda.h"
#include "mm_shared_nvidia.cuh"
#include "pmm-utils.cuh"

using namespace std;

extern "C" {

__device__
void _request_processing(
        int request_id, 
        volatile int* exit_signal,
        volatile int* request_signal,
        volatile int* request_counter,
        volatile int* request_ids, 
        volatile int** request_dest, 
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock){

    // SEMAPHORE
    acquire_semaphore((int*)lock, request_id);
    debug("mm: request recieved %d\n", request_id); 
    auto addr_id = request_ids[request_id];
    int request_status;
    
    switch (request_signal[request_id]){

        case MALLOC:
            if (addr_id == -1){
                addr_id = atomicAdd((int*)&request_counter[0], 1);
                request_ids[request_id] = addr_id;
            }else{
                assert(d_memory[addr_id] == NULL);
            }
            __threadfence();
            //d_memory[addr_id] = reinterpret_cast<volatile int*>
            //    (mm->malloc(4+request_mem_size[request_id]));
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
            request_status = d_memory[addr_id][0] - 1;
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
            //mm->free((void*)d_memory[addr_id]);
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
/*
__global__
void garbage_collector(volatile int** d_memory,
                       volatile int* requests_number, 
                       volatile int* request_counter,
                       volatile int* request_signal, 
                       volatile int* request_ids, 
                       volatile int* request_mem_size,
                       volatile int* lock,
                       volatile int* gc_started,
                       volatile int* exit_signal,
                       MemoryManagerType* mm){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
   
    gc_started[0] = 1;

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
*/

//producer
__global__
void mem_manager(volatile int* exit_signal, 
        volatile int* mm_started,
        volatile int* requests_number, 
        volatile int* request_counter,
        volatile int* request_signal, 
        volatile int* request_ids, 
        volatile int** request_dest,
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    mm_started[0] = 1;
    
    while (! exit_signal[0]){
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            __threadfence();
            if (request_signal[request_id] == MALLOC or 
                request_signal[request_id] == FREE){
                _request_processing(request_id, exit_signal, request_signal, 
                                    request_counter, request_ids, request_dest,
                                    d_memory, request_mem_size, lock);
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

    //int it = 0;
    // wait for success
    while (!exit_signal[0]){
        /*char* type_ = new char[10];
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
        }*/

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
void mul_app_test(int type, 
              volatile int* exit_signal,
              volatile int** d_memory, 
              volatile int* request_signal, 
              volatile int* request_mem_size,
              volatile int* request_id, 
              volatile int** request_dest, 
              volatile int* exit_counter, 
              volatile int* lock,
              int size_to_alloc,
              int iter_num,
              int MONO){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    //if (thid == 0) printf("start inner iteration\n");

    for (int i=0; i<iter_num; ++i){
        __threadfence();

        volatile int* new_ptr = NULL;
       
        if (MONO){
            request_id[thid] = thid;
            sharedABMultiply(A, B, C, N);
        }else{
            /*
            request(type, exit_signal, d_memory, &new_ptr, 
                    request_signal, request_mem_size, request_id, request_dest,
                    lock, size_to_alloc);*/
        }
        __threadfence();
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

void allocManaged(int** ptr, size_t size){
    GUARD_CU(cudaMallocManaged(ptr, size));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
}

void start_memory_manager(uint32_t mm_grid_size,
                          uint32_t block_size, 
                          CUcontext& mm_ctx,
                          int* exit_signal,
                          int* mm_started,
                          RequestType& requests){
    mem_manager<<<mm_grid_size, block_size>>>(exit_signal, mm_started,
            requests.requests_number, requests.request_counter, requests.request_signal, 
            requests.request_id, requests.request_dest,
            requests.d_memory, requests.request_mem_size, requests.lock);
    GUARD_CU(cudaPeekAtLastError());

}
/*
void start_garbage_collector(PerfMeasure& timing_gc, 
                          uint32_t gc_grid_size,
                          uint32_t block_size, 
                          CUcontext& gc_ctx,
                          int* exit_signal,
                          int* gc_started,
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
            gc_started,
            exit_signal,
#ifdef OUROBOROS__
            memory_manager.getDeviceMemoryManager()
#else
#ifdef HALLOC__
            &memory_manager
#endif
#endif
            );
    timing_gc.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());

}
*/
/*
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
            &memory_manager,
#endif
#endif
            requests.requests_number);

    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
    requests.free();
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}
*/

void start_application(int type, 
                       uint32_t grid_size,
                       uint32_t block_size, 
                       CUcontext& ctx,
                       int* exit_signal,
                       RequestType& requests,
                       int* exit_counter,
                       int size_to_alloc, 
                       int iter_num,
                       int mono, 
                       bool& kernel_complete){
    // Run application
    //GUARD_CU(cudaPeekAtLastError());
    auto kernel = mul_app_test;
    //printf("start kernel\n");
    kernel<<<grid_size, block_size>>>(type, exit_signal, requests.A, 
            requests.B, requests.C, requests.N,
            requests.request_signal, requests.request_mem_size, 
            requests.request_id, requests.request_dest, exit_counter, 
            requests.lock, size_to_alloc, iter_num, mono);

    //printf("kernel done, exit counter %d\n", exit_counter[0]);
    GUARD_CU(cudaPeekAtLastError());
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


void pmm_init(int mono, int kernel_iteration_num, int size_to_alloc, size_t* ins_size, 
              size_t num_iterations, int SMs, int* sm_app, int* sm_man, int* sm_gc, 
              int* matrix_mul, float* uni_req_per_sec, int* array_size){

    auto instant_size = *ins_size;

    printf("mono : %d\n", mono);

    CUcontext default_ctx;
    GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));

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

    int* gc_started;
    allocManaged(&gc_started, sizeof(uint32_t));

    int* mm_started;
    allocManaged(&mm_started, sizeof(uint32_t));

    int block_size = 1024;
    std::cout << "#requests\t" << "#sm app\t\t" << "#sm mm\t\t" << "#sm gc\t\t" << "#malloc and free per sec\n";

    if (mono){
        int app_grid_size = SMs;
        int requests_num = app_grid_size * block_size;
        sm_app[0] = app_grid_size;
        sm_man[0] = 0;
        matrix_mul[0] = requests_num;

        CUcontext app_ctx;
        CUdevice device;
        GUARD_CU((cudaError_t)cuDeviceGet(&device, 0));
        GUARD_CU((cudaError_t)cuCtxCreate(&app_ctx, 0, device));

        //CUexecAffinityParam_v1 app_param{
        //    CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)app_grid_size};
        //auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
        //GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx, &app_param, 1, affinity_flags, device));
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        PerfMeasure timing_malloc_app, app_synced;

        //printf("init done\n");
        for (int iteration = 0; iteration < num_iterations; ++iteration){

            *exit_signal = 0;
            *exit_counter = 0;
            RequestType requests;
            requests.init(requests_num);
            requests.memset();

            // Run APP (all threads do malloc)
            bool kernel_complete = false;
            std::thread app_thread{[&] {
                GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
                debug("start app\n");
                app_synced.startMeasurement();
                start_application(MUL, app_grid_size, block_size, app_ctx, 
                        exit_signal, requests, exit_counter, size_to_alloc, 
                        kernel_iteration_num, mono, kernel_complete);
                debug("app done, sync\n");
                GUARD_CU((cudaError_t)cuCtxSynchronize());
                app_synced.stopMeasurement();
                GUARD_CU(cudaPeekAtLastError());
                debug("done\n");
            }};

            //printf("join app\n");
            app_thread.join();
            //printf("app joined\n");

            *exit_signal = 1;

            //TODO: test
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
            continue;
        }

        GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));
        GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());

        auto app_synced_time   = app_synced.generateResult();
        uni_req_per_sec[0]   = (requests_num * 1000.0)/(app_synced_time.mean_/kernel_iteration_num);

        printf("  %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
                app_grid_size, 0, uni_req_per_sec[0]);

        *array_size = 1;

    }else{


        int it = 0;

        for (int app_grid_size=1; app_grid_size<SMs; ++app_grid_size){
            //for (int app_grid_size=1; app_grid_size<2; ++app_grid_size){

            for (int mm_grid_size=1; mm_grid_size<(SMs-app_grid_size); ++mm_grid_size){
                //for (int mm_grid_size=1; mm_grid_size<2; ++mm_grid_size){

                int requests_num{app_grid_size*block_size};

                //output
                sm_app[it] = app_grid_size;
                sm_man [it] = mm_grid_size;
                matrix_mul[it] = requests_num;

                CUexecAffinityParam_v1 app_param{
                    CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)app_grid_size};
                CUexecAffinityParam_v1 mm_param{
                    CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)mm_grid_size};
                
                auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
                CUcontext app_ctx, mm_ctx;
                CUdevice device;
                GUARD_CU((cudaError_t)cuDeviceGet(&device, 0));

                GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx, &app_param, 1, affinity_flags, device));
                GUARD_CU((cudaError_t)cuCtxCreate_v3(&mm_ctx, &mm_param, 1, affinity_flags, device));
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());

                //Timing variables
                PerfMeasure app_synced;

                for (int iteration = 0; iteration < num_iterations; ++iteration){

                    *exit_signal = 0;
                    *exit_counter = 0;
                    *mm_started = 0;
                    RequestType requests;
                    requests.init(requests_num);
                    requests.memset();

                    //GUARD_CU((cudaError_t)cuCtxGetCurrent(&default_ctx));
                    debug("start threads\n");

                    // Run Memory Manager (Presistent kernel)
                    std::thread mm_thread{[&] {
                        GUARD_CU((cudaError_t)cuCtxSetCurrent(mm_ctx));
                        //GUARD_CU((cudaError_t)cuCtxSynchronize());
                        debug("start mm\n");
                        start_memory_manager(mm_grid_size, block_size, mm_ctx,
                                exit_signal, mm_started, requests);
                        debug("mm done, sync\n");
                        GUARD_CU((cudaError_t)cuCtxSynchronize());
                        GUARD_CU(cudaPeekAtLastError());
                        debug("done\n");
                    }};

                    while (! *mm_started);
                    printf(".");
                    fflush(stdout);

                    // Run APP (all threads do malloc)
                    bool kernel_complete = false;
                    std::thread app_thread{[&] {
                        GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
                        //GUARD_CU((cudaError_t)cuCtxSynchronize());
                        debug("start app\n");
                        app_synced.startMeasurement();
                        start_application(MUL, app_grid_size, block_size, app_ctx, 
                                exit_signal, requests, exit_counter, size_to_alloc, 
                                kernel_iteration_num, mono, kernel_complete);
                        debug("app done, sync\n");
                        GUARD_CU((cudaError_t)cuCtxSynchronize());
                        app_synced.stopMeasurement();
                        GUARD_CU(cudaPeekAtLastError());
                        debug("done\n");
                    }};

                    //std::this_thread::sleep_for(std::chrono::seconds(1));

                    debug("join app\n");
                    app_thread.join();
                    debug("app joined\n");

                    *exit_signal = 1;

                    debug("join mm\n");
                    mm_thread.join();
                    debug("mm joined\n");

                    //TODO: test

                    GUARD_CU(cudaDeviceSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                }
                printf("\n");
                debug("done\n");

                GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));
                GUARD_CU((cudaError_t)cuCtxDestroy(mm_ctx));
                GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());

                // Output: the number of requests done per a second
                auto app_synced_time = app_synced.generateResult();
                uni_req_per_sec[it]  = (requests_num * 1000.0)/(app_synced_time.mean_/kernel_iteration_num);

                printf("  %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
                        app_grid_size, mm_grid_size, uni_req_per_sec[it]);

                ++it;
            }
            }
            *array_size = it;
    }
    
}

}
