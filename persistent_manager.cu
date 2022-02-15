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
        //volatile int* exit_signal,
        volatile int* request_signal,
        volatile int* request_counter,
        volatile int* request_ids, 
        //volatile int** request_dest, 
        //volatile int** d_memory,
        //volatile int* request_mem_size,
        volatile int* lock){

    // SEMAPHORE
    //acquire_semaphore((int*)lock, request_id);
    //int* ptr = (int*)(lock + request_id);
    //acquire_semaphore(ptr);
    acquire_semaphore((int*)(lock + request_id));
    //debug("mm: request recieved %d\n", request_id); 
    //auto addr_id = request_ids[request_id];

    if (request_signal[request_id] == MOCK){
        request_ids[request_id] = (request_ids[request_id] == -1 ? atomicAdd((int*)&request_counter[0], 1) :
        request_ids[request_id]);
        atomicExch((int*)&request_signal[request_id], DONE);
        __threadfence();
    }
    
    /*switch (request_signal[request_id]){

        case MOCK:
            //if (addr_id == -1){
                //addr_id = atomicAdd((int*)&request_counter[0], 1);
                //request_ids[request_id] = addr_id;
            if (request_ids[request_id] == -1){
                request_ids[request_id] = atomicAdd((int*)&request_counter[0], 1);
    //        }else{
      //          printf("error\n");
            }
            atomicExch((int*)&request_signal[request_id], request_done);
            __threadfence();
            break;

        default:
            break;
     //       printf("request processing fail\n");

    }*/

    //release_semaphore((int*)lock, request_id);
    release_semaphore((int*)(lock + request_id));
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
        volatile int** request_dest,
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock){
    //int thid = blockIdx.x * blockDim.x + threadIdx.x;

    mm_started[0] = 1;
    
    while (! exit_signal[0] ){
        for (int request_id = thid(); !exit_signal[0] && request_id < requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            __threadfence();
            if (request_signal[request_id] == MOCK){
                _request_processing(request_id, /*exit_signal,*/ request_signal, 
                                    request_counter, request_ids,/* request_dest,*/
                                    /*d_memory, request_mem_size,*/ lock);
                __threadfence();
                debug("mm: request done %d\n", request_id);
            }
        }
        __threadfence();
    }
}

__device__
void post_request(request_type type,
                  //volatile int** dest,
                  volatile int* lock,
                  //volatile int* request_mem_size,
                  //volatile int* request_id,
                  volatile int* request_signal
                  //,volatile int** request_dest,
                  //volatile int* exit_signal,
                  //int size_to_alloc
                  ){

    //int thid = blockDim.x * blockIdx.x + threadIdx.x;
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
}

__device__
void request_processed(request_type type,
                      volatile int* lock,
                      /*volatile int* request_id,
                      volatile int* exit_signal,
                      volatile int** d_memory,
                      volatile int** dest,*/
                      volatile int* request_signal
                      /*,volatile int** request_dest*/){
                      //int& req_id){
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
}

__device__
void request(request_type type,
        volatile int* exit_signal,
        //volatile int** d_memory,
        //volatile int** dest,
        volatile int* request_signal,
        //volatile int* request_mem_size, 
        //volatile int* request_id,
        //volatile int** request_dest,
        volatile int* lock
        //, int size_to_alloc
        ){

    // POST REQUEST: wait for success
    while (!exit_signal[0]){
        if (request_signal[thid()] == request_empty){
            post_request(type, /*dest,*/ lock, /*request_mem_size, 
                        request_id, */ request_signal /*,request_dest, 
                        exit_signal, size_to_alloc*/);
            break;
        }
        __threadfence();
    }
    ///__threadfence();

    // REQUEST PROCESSED
    // int it = 0;
    // wait for success
    while (!exit_signal[0]){
        if (request_signal[thid()] == request_done){
            request_processed(type, lock,/* request_id, exit_signal, d_memory, 
                        dest,*/ request_signal/*, request_dest*/);//, -1);
            break;
        }
        __threadfence();
    }
}


/*
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
}*/

}
