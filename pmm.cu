#include <iostream>
#include <string>
#include <cassert>
#include <algorithm>
#include <thread>
#include <chrono>
#include <any>

#include "PerformanceMeasure.cuh"
#include "cuda.h"
#include "pmm-utils.cuh"
#include "persistent_manager.cu"

using namespace std;

extern "C" {

void createStreams(cudaStream_t& mm_stream, 
                   cudaStream_t& app_stream){
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

//consumer
__global__
void mul_app_test(request_type type, 
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

    for (int i=0; i<iter_num; ++i){
        __threadfence();

        volatile int* new_ptr = NULL;
       
        if (MONO){
            request_id[thid] = thid;
        }else{
            request(type, exit_signal, d_memory, &new_ptr, 
                    request_signal, request_mem_size, request_id, 
                    request_dest, lock, size_to_alloc);
        }
        __threadfence();
    }
    
    atomicAdd((int*)&exit_counter[0], 1);

    __threadfence();
    //printf("exit counter[%d] = %d\n", thid, exit_counter[0]);
}

void start_application(request_type type, 
                       uint32_t grid_size,
                       uint32_t block_size, 
                       CUcontext& ctx,
                       volatile int* exit_signal,
                       RequestType& requests,
                       volatile int* exit_counter,
                       int size_to_alloc, 
                       int iter_num,
                       int mono, 
                       bool& kernel_complete){
    // Run application
    //GUARD_CU(cudaPeekAtLastError());
    auto kernel = mul_app_test;
    
    //printf("start kernel\n");
    kernel<<<grid_size, block_size>>>(type, exit_signal, requests.d_memory,
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
    cudaStream_t mm_stream, app_stream;
    createStreams(mm_stream, app_stream);

    int* exit_signal;
    allocManaged(&exit_signal, sizeof(int32_t));

    int* exit_counter;
    allocManaged(&exit_counter, sizeof(uint32_t));

    int* mm_started;
    allocManaged(&mm_started, sizeof(uint32_t));

    int block_size = 1024;
    std::cout << "#requests\t" << "#sm app\t\t" << "#sm mm\t\t" 
                << "#sm gc\t\t" << "#malloc and free per sec\n";

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
                start_application((request_type)MATRIX_MUL, app_grid_size, block_size, app_ctx, 
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

        auto app_synced_time = app_synced.generateResult();
        uni_req_per_sec[0]   = (requests_num * 1000.0)/
                                    (app_synced_time.mean_/kernel_iteration_num);

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
                    CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                                                    (unsigned int)app_grid_size};
                CUexecAffinityParam_v1 mm_param{
                    CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, 
                                                    (unsigned int)mm_grid_size};
                
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
                        start_application((request_type)MATRIX_MUL, app_grid_size, block_size, app_ctx, 
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
