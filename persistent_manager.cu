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
/*
__device__
void _request_processing(
        int request_id, 
        volatile int* request_signal,
        volatile int* request_counter,
        volatile int* request_ids, 
        volatile int* lock){

    // SEMAPHORE
    acquire_semaphore((int*)(lock + request_id));
    if (request_signal[request_id] == MOCK){
        if (request_ids[request_id] == -1){
            request_ids[request_id] = atomicAdd((int*)&request_counter[0], 1);
        }
        atomicExch((int*)&request_signal[request_id], DONE);
        __threadfence();
    }
    release_semaphore((int*)(lock + request_id));
    // SEMAPHORE
}*/

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
    uint32_t ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    auto es = exit_signal[0];
    if (blockIdx.x > 30)
        printf("1: memory manager: block id %d/%d, smid %d, exit_signal = %d\n", blockIdx.x, gridDim.x, ret, es);

    while (! exit_signal[0] ){
        asm("mov.u32 %0, %smid;" : "=r"(ret));
        if (blockIdx.x > 30)
            printf("2: memory manager: block id %d/%d, smid %d\n", blockIdx.x, gridDim.x, ret);
        for (int request_id = thid(); !exit_signal[0] && 
                request_id < requests_number[0]; 
                request_id += blockDim.x*gridDim.x){
                uint32_t ret;
                asm("mov.u32 %0, %smid;" : "=r"(ret));
                if (blockIdx.x > 30)
                    printf("3: memory manager: request id %d/%d, smid %d\n", request_id, requests_number[0], ret);
        }
        __threadfence();
        break;
    }
}
/*
__device__
void post_request(request_type type,
                  volatile int* lock,
                  volatile int* request_signal){

    debug("request %d, block %d\n", thid(), blockIdx.x);
    
    __threadfence();
    // SEMAPHORE
    //acquire_semaphore((int*)lock, thid());
    //int* ptr = (int*)(lock + thid());
    acquire_semaphore((int*)(lock + thid()));
    // SIGNAL update
    atomicExch((int*)&request_signal[thid()], type);
    //release_semaphore((int*)lock, thid());
    release_semaphore((int*)(lock + thid()));

    __threadfence();
    // SEMAPHORE
}*/
/*
__device__
void request_processed(request_type type,
                      volatile int* lock,
                      volatile int* request_signal){
    //int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // SEMAPHORE
    __threadfence();
    //acquire_semaphore((int*)lock, thid());
    //int* ptr = (int*)(lock + thid());
    acquire_semaphore((int*)(lock + thid()));
    switch (type){
        case MOCK:
            //req_id = request_id[thid()];
            break;
        default:
            //printf("error\n");
            break;
    }
    // SIGNAL update
    request_signal[thid()] = request_empty;
    //release_semaphore((int*)lock, thid());
    release_semaphore((int*)(lock + thid()));
    //debug("request %d, block %d done\n", thid(), blockIdx.x);
    __threadfence();
    // SEMAPHORE
}*/

__device__
void request(request_type type,
        volatile int* exit_signal,
        volatile int* request_signal,
        volatile int* lock
        ){

    // POST REQUEST: wait for success
    while (!exit_signal[0]){
        uint32_t ret;
        if (thid() > 30){
            asm("mov.u32 %0, %smid;" : "=r"(ret));
            printf("requested by %d/%d, smid %d\n", thid(), blockDim.x*gridDim.x, ret);
        }
        break;
        /*if (request_signal[thid()] == request_empty){
            //post_request(type, lock, request_signal);
        }
        __threadfence();*/
    }

    // REQUEST PROCESSED
    // int it = 0;
    // wait for success
    while (!exit_signal[0]){
        break;
        /*
        if (request_signal[thid()] == request_done){
            //request_processed(type, lock, request_signal);
            break;
        }
        __threadfence();*/
    }
}

}
