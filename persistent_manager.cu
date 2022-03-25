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

using namespace std;

extern "C" {

__device__
void _request_processing(
        int request_id, 
        volatile int* request_signal,
        volatile int* request_counter,
        volatile int* request_ids, 
        volatile int* lock){

    // SEMAPHORE
    acquire_semaphore(lock + request_id);
    auto rs = request_signal[request_id];
    if (rs == MOCK){
        if (request_ids[request_id] == -1){
            request_ids[request_id] = atomicAdd((int*)&request_counter[0], 1);
        }
        //printf("request done %d\n", request_id);
        atomicExch((int*)&request_signal[request_id], DONE);
        __threadfence();
    }else{
        //printf("error: request_signal = %d\n", rs);
    }
    release_semaphore(lock + request_id);
    // SEMAPHORE
}

//producer
__global__
void mem_manager(volatile int* exit_signal, 
        volatile int* mm_started,
        volatile int* requests_number, 
        volatile int* request_counter,
        volatile int* request_signal, 
        volatile int* request_ids, 
        volatile int* lock){
    mm_started[0] = 1;
        auto es = exit_signal[0];
    //if (blockIdx.x > 30)
        int first_comm = 1;
        int second_comm = 1;
    while (! exit_signal[0] ){
           // break;
        for (int request_id = thid(); !exit_signal[0] && 
                request_id < requests_number[0]; 
                request_id += blockDim.x*gridDim.x){
            __threadfence();
            auto rs = request_signal[request_id];

            if (first_comm){
                printf("MEM MANAGER(before): blockIdx.x=%d/%d, exit_signal = %d, request_signal=%d, sm = %d, lane = %d, warp = %d\n", blockIdx.x, gridDim.x, es, rs, sm_id(), lane_id(), warp_id());
                first_comm = 0;
            }
            if (rs == MOCK){
                _request_processing(request_id, request_signal, 
                                    request_counter, request_ids, lock);
                __threadfence();
                debug("mm: request done %d\n", request_id);

            if (second_comm){
                printf("MEM MANAGER(after): blockIdx.x=%d/%d, exit_signal = %d, request_signal=%d, sm = %d, lane = %d, warp = %d\n", blockIdx.x, gridDim.x, es, rs, sm_id(), lane_id(), warp_id());
                second_comm = 0;
            }

            }
        }
        __threadfence();
    }
}

__device__
void post_request(request_type type,
                  volatile int* lock,
                  volatile int* request_signal
                  ){
    //printf("request %d, block %d\n", thid(), blockIdx.x);
    
    __threadfence();
    // SEMAPHORE
    acquire_semaphore(lock + thid());
    // SIGNAL update
   // request_signal[thid()] = type;
    atomicExch((int*)&request_signal[thid()], type);
    release_semaphore(lock + thid());

    __threadfence();
    // SEMAPHORE
}

__device__
void request_processed(request_type type,
                      volatile int* lock,
                      volatile int* request_signal){
    // SEMAPHORE
    __threadfence();
    acquire_semaphore(lock + thid());
    // SIGNAL update
    //request_signal[thid()] = request_empty;
    atomicExch((int*)&request_signal[thid()], request_empty);
    release_semaphore(lock + thid());
    // SEMAPHORE
}

__device__
void request(request_type type,
        volatile int* exit_signal,
        volatile int* request_signal,
        volatile int* lock
        ){

    //printf("request %d\n", thid());
    //debug("request %d\n", thid());
    // POST REQUEST: wait for success
    while (!exit_signal[0]){
        //printf("%d/%d i am gonna post request\n", thid(), blockIdx.x);
        if (request_signal[thid()] == request_empty){
            //printf("post request %d\n", thid());
            post_request(type, lock, request_signal);
            break;
        }
        __threadfence();
    }

    auto es = exit_signal[0];
    printf("APP: blockIdx.x=%d/%d, exit_signal = %d, sm = %d, lane = %d, warp = %d\n", blockIdx.x, gridDim.x, es, sm_id(), lane_id(), warp_id());
 
    //printf("%d/%d waiting to be done\n", thid(), blockIdx.x);
    // REQUEST PROCESSED
    // wait for success
    while (!exit_signal[0]){
        auto rs = request_signal[thid()];
        if (rs == request_done){
            //printf("request processed %d\n", thid());
            request_processed(type, lock, request_signal);
            break;
        }
        __threadfence();
    }
}

}
