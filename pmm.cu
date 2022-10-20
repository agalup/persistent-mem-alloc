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
        RequestType requests,
       /* volatile int* request_signal,
        volatile int* request_counter,
        volatile int* request_ids, 
        volatile int** request_dest, */
        MemoryManagerType* mm
       /*, volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock*/){
    debug("request processing!\n");

    // SEMAPHORE
    acquire_semaphore((int*)requests.lock, request_id);
    debug("MEM MANAGER %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, request_id, blockIdx.x, warp_id(), lane_id(), sm_id());
    //debug("mm: request recieved %d\n", request_id); 
    auto addr_id = requests.request_id[request_id];
    int request_status;
    
    switch (requests.request_signal[request_id]){

        case MALLOC:
            if (addr_id == -1){
                addr_id = atomicAdd((int*)&requests.request_counter[0], 1);
                requests.request_id[request_id] = addr_id;
            }else{
                assert(requests.d_memory[addr_id] == NULL);
            }
            __threadfence();
            requests.d_memory[addr_id] = reinterpret_cast<volatile int*>
                (mm->malloc(4+requests.request_mem_size[request_id]));
            __threadfence();
            assert(requests.d_memory[addr_id]);
            requests.d_memory[addr_id][0] = 0;
            requests.request_dest[request_id] = &requests.d_memory[addr_id][1];
            atomicExch((int*)&requests.request_signal[request_id], request_done);
            break;

        case FREE:
            assert(requests.d_memory[addr_id]);
            if (requests.d_memory[addr_id][0] != 0)
                printf("d_memory{%d} = %d\n", addr_id, requests.d_memory[addr_id][0]);
            assert(requests.d_memory[addr_id][0] == 0);
            request_status = requests.d_memory[addr_id][0] - 1;
            requests.d_memory[addr_id][0] -= 1;
            requests.request_dest[request_id] = NULL;
            assert(requests.d_memory[addr_id][0] == -1);
            if (request_status < 0){
                atomicExch((int*)&requests.request_signal[request_id], request_gc);
            }else{
                assert(1);
                printf("should not be here!\n");
                atomicExch((int*)&requests.request_signal[request_id], request_done);
            }
            break;

        case GC:
            assert(requests.d_memory[addr_id]);
            assert(requests.d_memory[addr_id][0] == -1);
            __threadfence();
            mm->free((void*)requests.d_memory[addr_id]);
            __threadfence();
            requests.d_memory[addr_id] = NULL;
            atomicExch((int*)&requests.request_signal[request_id], request_done);
            break;

        default:
            printf("request processing fail\n");

    }
    release_semaphore((int*)requests.lock, request_id);
    // SEMAPHORE
}

/*__global__
void garbage_collector(
                       volatile int* exit_signal,
                       volatile int* gc_started,
                       volatile int* requests_number, 
                       volatile int* request_counter,
                       volatile int* request_signal, 
                       volatile int* request_ids, 
                       volatile int* request_mem_size,
                       volatile int** request_dest,
                       volatile int** d_memory,
                       volatile int* lock,
                       MemoryManagerType* mm
                       ){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    gc_started[0] = 1;
    while (! exit_signal[0]){
        debug("hello gc! %d\n", thid);
        assert(requests_number);
        assert(request_signal);
        for (int request_id = thid; !exit_signal[0] && 
                request_id < requests_number[0]; 
                request_id += blockDim.x*gridDim.x){
            __threadfence();
            if ((request_signal[request_id]) == GC){
                _request_processing(request_id, exit_signal, 
                                    request_signal,
                                    request_counter, 
                                    request_ids, 
                                    request_dest,
                                    mm, d_memory, 
                                    request_mem_size, 
                                    lock);
                __threadfence();
            }
        }
        __threadfence();
    }
}*/

__global__
void garbage_collector(
                       volatile int* exit_signal,
                       volatile int* gc_started,
                       RequestType requests,
                       MemoryManagerType* mm
                       ){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    gc_started[0] = 1;
    while (! exit_signal[0]){
        debug("hello gc! %d\n", thid);
        //assert(requests_number);
        //assert(request_signal);
        for (int request_id = thid; !exit_signal[0] && 
                request_id < requests.requests_number[0]; 
                request_id += blockDim.x*gridDim.x){
            __threadfence();
            if ((requests.request_signal[request_id]) == GC){
                _request_processing(request_id, exit_signal, 
                                    requests, mm);
                __threadfence();
            }
        }
        __threadfence();
    }
}



//producer
__global__
void mem_manager(volatile int* exit_signal, 
                volatile int* mm_started,
                RequestType requests,
                /*volatile int* requests_number, 
                volatile int* request_counter,
                volatile int* request_signal, 
                volatile int* request_ids, 
                volatile int* request_mem_size,
                volatile int** request_dest,
                volatile int** d_memory,
                volatile int* lock,*/
                MemoryManagerType* mm
                ){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    mm_started[0] = 1;
    while (! exit_signal[0]){
        debug("hello mm %d, request no %d!\n", thid, requests_number[0]);
        for (int request_id = thid; !exit_signal[0] && 
                request_id < requests.requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            __threadfence();
            if ((requests.request_signal[request_id]) == MALLOC or 
                (requests.request_signal[request_id]) == FREE){
                _request_processing(request_id, exit_signal, 
                                    requests, mm);
                /*
                                    requests.request_signal, 
                                    requests.request_counter, 
                                    requests.request_id, 
                                    requests.request_dest,
                                    mm, requests.d_memory, 
                                    requests.request_mem_size, 
                                    requests.lock);*/
                __threadfence();
                debug("mm: request done %d\n", request_id);
            }
        }
        /*
        for (int request_id = thid; !exit_signal[0] && 
                request_id < requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            __threadfence();
            if ((request_signal[request_id]) == MALLOC or 
                (request_signal[request_id]) == FREE){
                _request_processing(request_id, exit_signal, 
                                    request_signal, 
                                    request_counter, 
                                    request_ids, 
                                    request_dest,
                                    mm, d_memory, 
                                    request_mem_size, 
                                    lock);
                __threadfence();
                debug("mm: request done %d\n", request_id);
            }
        }*/
        __threadfence();
    }
}

/*__device__
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
    debug("APP %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, thid, blockIdx.x, warp_id(), lane_id(), sm_id());
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
    debug("APP %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, thid, blockIdx.x, warp_id(), lane_id(), sm_id());
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
*/
__device__
void post_request(request_type type,
                  volatile int* exit_signal,
                  RequestType& requests,
                  volatile int** dest,
                  int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    __threadfence();
    // SEMAPHORE
    acquire_semaphore((int*)requests.lock, thid);
    if (type == MALLOC){
        requests.request_mem_size[thid] = size_to_alloc;
    }
    // SIGNAL update
    atomicExch((int*)&requests.request_signal[thid], type);
    debug("APP %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, thid, blockIdx.x, warp_id(), lane_id(), sm_id());
    release_semaphore((int*)requests.lock, thid);
    __threadfence();
    // SEMAPHORE
}

__device__
void request_processed(request_type type,
                      volatile int* exit_signal,
                      RequestType& requests,
                      volatile int** dest,
                      int& req_id){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // SEMAPHORE
    __threadfence();
    acquire_semaphore((int*)requests.lock, thid);
    debug("APP %s: thid %d, block ID %d, warp ID %d, lane ID %d, sm ID %d\n", __FUNCTION__, thid, blockIdx.x, warp_id(), lane_id(), sm_id());
    switch (type){
        case MALLOC:
            req_id = requests.request_id[thid];
            if (req_id >= 0 && !exit_signal[0]) {
                *dest = requests.request_dest[thid];
                assert(requests.d_memory[req_id] != NULL);
                if (requests.d_memory[req_id][0] != 0)
                    printf("d_memory[%d] = %d\n", req_id, requests.d_memory[req_id][0]);
                //assert(d_memory[req_id][0] == 0);
                assert(*dest != NULL);
                assert(requests.request_dest[thid] == *dest);
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
    requests.request_signal[thid] = request_empty;
    release_semaphore((int*)requests.lock, thid);
    __threadfence();
    // SEMAPHORE
}
/*
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
        //char* type_ = new char[10];
        //char* state = new char[10];
        //if (++it > 1000){
        //    if (type == MALLOC) type_ = "MALLOC"; else type_ = "FREE";
        //    switch (request_signal[thid]){
        //        case request_empty:  state = "EMPTY";  break;
        //        case request_done:   state = "DONE";   break;
        //        case request_malloc: state = "MALLOC"; break;
        //        case request_free:   state = "FREE";   break;
        //        case request_gc:     state = "GC";     break;
        //    }
        //    printf("thid %d, current state %s\n", thid, state);
        //}

        if (request_signal[thid] == request_done){
            request_processed(type, lock, request_id, exit_signal, d_memory, 
                        dest, request_signal, request_dest, req_id);
            break;
        }
        __threadfence();
    }
}
*/

__device__
void request(request_type type,
        volatile int* exit_signal,
        RequestType& requests,
        volatile int** dest,
        int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int req_id = -1;
    // wait for success
    while (!exit_signal[0]){
        if (requests.request_signal[thid] == request_empty){
            post_request(type, exit_signal, requests, dest, 
                         size_to_alloc);
            break;
        }
        __threadfence();
    }
    __threadfence();

    // wait for success
    while (!exit_signal[0]){
        if (requests.request_signal[thid] == request_done){
            request_processed(type, exit_signal, requests, dest, 
                              req_id);
            break;
        }
        __threadfence();
    }
}

//consumer
__global__
void mono_app_test(//volatile int* exit_signal,
        volatile int** d_memory, 
        //RequestType& requests, 
        //volatile int* request_signal, 
        //volatile int* request_mem_size,
        //volatile int* request_id, 
        //volatile int** request_dest, 
        volatile int* exit_counter, 
        //volatile int* lock,
        int* size_to_alloc,
        int* iter_num,
        //int MONO,
        MemoryManagerType* mm){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;

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

/*//consumer
__global__
void app_test(volatile int* exit_signal,
        volatile int** d_memory, 
        volatile int* request_signal, 
        volatile int* request_mem_size,
        volatile int* request_id, 
        volatile int** request_dest, 
        volatile int* exit_counter, 
        volatile int* lock,
        int* size_to_alloc,
        int* iter_num,
        int MONO,
        MemoryManagerType* mm){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i=0; i<iter_num[0]; ++i){
        __threadfence();

        volatile int* new_ptr = NULL;

        request((request_type)MALLOC, exit_signal, d_memory, &new_ptr, 
                request_signal, request_mem_size, request_id, request_dest,
                lock, size_to_alloc[0]);
        new_ptr[0] = thid;

        __threadfence();

        assert(d_memory[request_id[thid]]);
        //int value = d_memory[request_id[thid]][0];
        //if (value != 0) printf("val = %d\n", value);
        assert(new_ptr[0] == thid);

        __threadfence();

        request((request_type)FREE, exit_signal, d_memory, &new_ptr,
                request_signal, request_mem_size, request_id, request_dest,
                lock, size_to_alloc[0]);

        __threadfence();
    }
    atomicAdd((int*)&exit_counter[0], 1);
    __threadfence();

}
*/

//consumer
__global__
void app_test(volatile int* exit_signal,
        volatile int* exit_counter, 
        RequestType requests,
        int* size_to_alloc,
        int* iter_num,
        int MONO,
        MemoryManagerType* mm){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i=0; i<iter_num[0]; ++i){
        __threadfence();

        volatile int* new_ptr = NULL;

        request((request_type)MALLOC, exit_signal, requests, &new_ptr, 
                 size_to_alloc[0]);
        new_ptr[0] = thid;

        __threadfence();

        assert(requests.d_memory[requests.request_id[thid]]);
        //int value = d_memory[request_id[thid]][0];
        //if (value != 0) printf("val = %d\n", value);
        assert(new_ptr[0] == thid);

        __threadfence();

        request((request_type)FREE, exit_signal, requests, &new_ptr,
                 size_to_alloc[0]);

        __threadfence();
    }
    atomicAdd((int*)&exit_counter[0], 1);
    __threadfence();

}
/*
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
*/
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

void start_memory_manager(PerfMeasure& timing_mm, 
                          uint32_t mm_grid_size,
                          uint32_t block_size, 
                          CUcontext& mm_ctx,
                          volatile int* exit_signal,
                          volatile int* mm_started,
                          RequestType& requests,
                          MemoryManagerType& memory_manager){
    timing_mm.startMeasurement();
  
    auto dev_mm = memory_manager.getDeviceMemoryManager();
    void *args[] = {&exit_signal, &mm_started, &requests, &dev_mm};
    
    //GUARD_CU(cudaLaunchCooperativeKernel((void*)mem_manager, mm_grid_size, block_size, args));

    GUARD_CU(cudaLaunchKernel((void*)mem_manager, mm_grid_size, block_size, args));
    GUARD_CU((cudaError_t)cudaGetLastError());
    GUARD_CU(cudaPeekAtLastError());

    timing_mm.stopMeasurement();
}

void start_garbage_collector(PerfMeasure& timing_gc, 
                          uint32_t gc_grid_size,
                          uint32_t block_size, 
                          CUcontext& gc_ctx,
                          volatile int* exit_signal,
                          volatile int* gc_started,
                          RequestType& requests,
                          MemoryManagerType& memory_manager){
    timing_gc.startMeasurement();
    
    auto dev_mm = memory_manager.getDeviceMemoryManager();
    void *args[] = {&exit_signal, &gc_started, &requests, &dev_mm};
    
    //GUARD_CU(cudaLaunchCooperativeKernel((void*)garbage_collector, gc_grid_size, block_size, args));

    GUARD_CU(cudaLaunchKernel((void*)garbage_collector, gc_grid_size, block_size, args));
    GUARD_CU((cudaError_t)cudaGetLastError());
    GUARD_CU(cudaPeekAtLastError());
    
    timing_gc.stopMeasurement();
}

void clean_memory(uint32_t grid_size,
                  uint32_t block_size, 
                  RequestType& requests,
                  MemoryManagerType& memory_manager,
                  volatile int* exit_signal){

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
                       PerfMeasure& timing_sync, 
                       uint32_t grid_size,
                       uint32_t block_size, 
                       CUcontext& ctx,
                       volatile int* exit_signal,
                       RequestType& requests,
                       volatile int* exit_counter,
                       int* dev_size_to_alloc, 
                       int* dev_iter_num,
                       int mono, 
                       bool& kernel_complete,
                       MemoryManagerType& memory_manager){
    assert(requests.d_memory);
    fflush(stdout);
    if (mono == MPS_mono){
        printf("start application: MPS mono!\n");
        /*int* dev_size_to_alloc, *dev_iter_num;
        GUARD_CU(cudaMalloc((void**)&dev_size_to_alloc, sizeof(int)));
        GUARD_CU(cudaMalloc((void**)&dev_iter_num, sizeof(int)));
        GUARD_CU(cudaMemset(dev_size_to_alloc, size_to_alloc, sizeof(int)));
        GUARD_CU(cudaMemset(dev_iter_num, iter_num, sizeof(int)));*/
        auto dev_mm = memory_manager.getDeviceMemoryManager();
        void* args[] = {&requests.d_memory, &exit_counter, &dev_size_to_alloc, &dev_iter_num, &dev_mm};

        GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        GUARD_CU(cudaLaunchKernel((void*)mono_app_test, dim3(grid_size, 1, 1), dim3(block_size, 1, 1), args, 0, 0));
        /*
        mono_app_test<<<grid_size, block_size>>>(
            requests.d_memory, exit_counter, size_to_alloc, iter_num, 
            memory_manager.getDeviceMemoryManager());
        */
        GUARD_CU((cudaError_t)cuCtxSynchronize());
        GUARD_CU(cudaPeekAtLastError());
        timing_sync.stopMeasurement();
        GUARD_CU(cudaProfilerStop());
    }else{
        debug("start applications: type %d\n", type);
        auto dev_mm = memory_manager.getDeviceMemoryManager();

        void* args[] = {&exit_signal, &exit_counter, &requests, 
                        &dev_size_to_alloc, &dev_iter_num, &mono, 
                        &dev_mm};
       /*void* args[] = {&exit_signal, &requests.d_memory, 
                        &requests.request_signal,
                        &requests.request_mem_size, 
                        &requests.request_id, 
                        &requests.request_dest, &exit_counter, 
                        &requests.lock, &dev_size_to_alloc, 
                        &dev_iter_num, &mono, &dev_mm};*/

        //GUARD_CU(cudaProfilerStart());
        timing_sync.startMeasurement();
        GUARD_CU(cudaLaunchKernel((void*)app_test, grid_size, block_size, args, 0, 0));
        //GUARD_CU(cudaLaunchCooperativeKernel((void*)app_test, grid_size, block_size, kernelArgs));
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


void simple_monolithic_app(int mono, int kernel_iteration_num, int size_to_alloc, size_t* ins_size, 
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
    
    volatile int* exit_signal;
    allocManaged(&exit_signal, sizeof(int32_t));

    volatile int* exit_counter;
    allocManaged(&exit_counter, sizeof(uint32_t));

    int* dev_size_to_alloc;
    int* dev_kernel_iteration_num;
    GUARD_CU(cudaMallocManaged(&dev_size_to_alloc, sizeof(int)));
    GUARD_CU(cudaMallocManaged(&dev_kernel_iteration_num, sizeof(int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
   
    *dev_size_to_alloc = size_to_alloc;
    *dev_kernel_iteration_num = kernel_iteration_num;
    
    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, 0));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)dev_size_to_alloc, sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)dev_kernel_iteration_num, sizeof(int), device, NULL));

    int block_size = 1024;
    int app_grid_size = SMs;
    int requests_num = app_grid_size * block_size;
    
    sm_app[0] = app_grid_size;
    sm_mm[0] = 0;
    sm_gc[0] = 0;
    allocs[0] = requests_num;

    cudaStream_t app_stream;
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());

    PerfMeasure /*timing_malloc_app, */malloc_total_sync;
    for (int iteration = 0; iteration < num_iterations; ++iteration){
        *exit_counter = 0;
        volatile int** d_memory{nullptr};
        GUARD_CU(cudaMalloc(&d_memory, requests_num * sizeof(volatile int*)));
        GUARD_CU(cudaPeekAtLastError());
        malloc_total_sync.startMeasurement();
        mono_app_test<<<app_grid_size, block_size, 0, app_stream>>>(d_memory, 
                                exit_counter, dev_size_to_alloc, dev_kernel_iteration_num, 
                                memory_manager.getDeviceMemoryManager());
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
    //uni_req_per_sec[0]   = (requests_num * 1000.0)/(malloc_total_sync_res.mean_/total_iters);
    uni_req_per_sec[0] = (requests_num * 2000.0)/malloc_total_sync_res.mean_;

    printf("#measurements %d, mean %.2lf, #total iters %lu\n", malloc_total_sync_res.num_, malloc_total_sync_res.mean_, total_iters);

    printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
            app_grid_size, 0, 0, uni_req_per_sec[0]);

    *array_size = 1;

}

void mps_monolithic_app(int mono, int kernel_iteration_num, int size_to_alloc, 
            size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
            int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
            int* array_size){

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
    
    volatile int* exit_signal;
    allocManaged(&exit_signal, sizeof(int32_t));

    volatile int* exit_counter;
    allocManaged(&exit_counter, sizeof(uint32_t));

    CUcontext app_ctx;
    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, 0));
    GUARD_CU((cudaError_t)cuCtxCreate(&app_ctx, 0, device));

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
    int block_size = 1024;
    int requests_num = app_grid_size * block_size;
    sm_app[0] = app_grid_size;
    sm_mm[0] = 0;
    sm_gc[0] = 0;
    allocs[0] = requests_num;

    PerfMeasure /*timing_malloc_app, */malloc_total_sync;

    for (int iteration = 0; iteration < num_iterations; ++iteration){
        //printf("iter %d, requests_num %d\n", iteration, requests_num);

        *exit_signal = 0;
        *exit_counter = 0;

        GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_signal, sizeof(int), device, NULL));
        GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_counter, sizeof(int), device, NULL));

        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());

        RequestType requests;
        requests.init(requests_num);
        requests.memset();

        // Run APP (all threads do malloc)
        bool kernel_complete = false;
        std::thread app_thread{[&] {
            GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
            //GUARD_CU((cudaError_t)cuCtxSynchronize());
            debug("start app\n");
            //malloc_total_sync.startMeasurement();
            start_application(MALLOC, malloc_total_sync, 
                    app_grid_size, block_size, app_ctx, exit_signal,
                    requests, exit_counter, dev_size_to_alloc, 
                    dev_kernel_iteration_num, mono, kernel_complete, memory_manager);
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
            clean_memory(app_grid_size, block_size, requests, memory_manager, exit_signal);
            continue;
        }

        *exit_signal = 1;

        clean_memory(app_grid_size, block_size, requests, memory_manager, exit_signal);
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(cudaPeekAtLastError());
    }

    GUARD_CU((cudaError_t)cuCtxDestroy(app_ctx));
    GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    auto malloc_total_sync_res = malloc_total_sync.generateResult();
    auto total_iters = kernel_iteration_num*num_iterations;
    //uni_req_per_sec[0] = (requests_num * 1000.0)/(malloc_total_sync_res.mean_/total_iters);
    uni_req_per_sec[0] = (requests_num * 2000.0)/malloc_total_sync_res.mean_;

    printf("#measurements %d, mean %.2lf, #total iters %lu\n", malloc_total_sync_res.num_, malloc_total_sync_res.mean_, total_iters);

    printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
            app_grid_size, 0, 0, uni_req_per_sec[0]);

    *array_size = 1;

}

void mps_app(int mono, int kernel_iteration_num, int size_to_alloc, 
        size_t* ins_size, size_t num_iterations, int SMs, int* sm_app, 
        int* sm_mm, int* sm_gc, int* allocs, float* uni_req_per_sec, 
        int* array_size){

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

    CUdevice device;
    GUARD_CU((cudaError_t)cuDeviceGet(&device, 0));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t gc_stream, mm_stream, app_stream;
    createStreams(gc_stream, mm_stream, app_stream);

    volatile int* exit_signal;  allocManaged(&exit_signal, sizeof(int32_t));
    volatile int* exit_counter; allocManaged(&exit_counter, sizeof(uint32_t));
    volatile int* gc_started;   allocManaged(&gc_started, sizeof(uint32_t));
    volatile int* mm_started;   allocManaged(&mm_started, sizeof(uint32_t));

    int* dev_size_to_alloc;        allocManaged_(&dev_size_to_alloc, sizeof(int));
    int* dev_kernel_iteration_num; allocManaged_(&dev_kernel_iteration_num, 
                                                    sizeof(int));
    //GUARD_CU(cudaMallocManaged(&dev_size_to_alloc, sizeof(int)));
    //GUARD_CU(cudaMallocManaged(&dev_kernel_iteration_num, sizeof(int)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
   
    *dev_size_to_alloc = size_to_alloc;
    *dev_kernel_iteration_num = kernel_iteration_num;
    
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)dev_size_to_alloc, 
                                                sizeof(int), device, NULL));
    GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)dev_kernel_iteration_num, 
                                                sizeof(int), device, NULL));

    int it = 0;

    int block_size = 1024;

    //SMs -= 10;

    for (int app_grid_size=1; app_grid_size<SMs; ++app_grid_size){
        //for (int app_grid_size=27; app_grid_size<28; ++app_grid_size){
    //for (int app_grid_size=1; app_grid_size<2; ++app_grid_size){
        for (int mm_grid_size=1; mm_grid_size<(SMs-app_grid_size); ++mm_grid_size){
            //for (int mm_grid_size=8; mm_grid_size<9; ++mm_grid_size){
        //for (int mm_grid_size=1; mm_grid_size<2; ++mm_grid_size){

            int gc_grid_size = SMs - app_grid_size - mm_grid_size;
            //int gc_grid_size = 1;
            if (gc_grid_size <= 0) continue;

            int requests_num{app_grid_size * block_size};

            debug("SMs: app %d, mm %d, gc %d, total %d\n", app_grid_size, mm_grid_size, gc_grid_size, SMs);
            debug("requests_num %d\n", requests_num);
            fflush(stdout);

            //output
            sm_app[it] = app_grid_size;
            sm_mm [it] = mm_grid_size;
            sm_gc [it] = gc_grid_size;
            allocs[it] = requests_num;
    
            //int mul = 1;
            int app_numBlocksPerSm = 1;// 0;
            int gc_numBlocksPerSm =  1;//0;
            int mm_numBlocksPerSm =  1;//0;

            GUARD_CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&app_numBlocksPerSm, app_test, block_size, 0));
            GUARD_CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gc_numBlocksPerSm, mem_manager, block_size, 0));
            GUARD_CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mm_numBlocksPerSm, garbage_collector, block_size,
            0));

            printf("num blocks per sm by cudaOccMaxActBlPerSM: app %d, mm %d, gc %d\n", 
            app_numBlocksPerSm, gc_numBlocksPerSm, mm_numBlocksPerSm);
            fflush(stdout);

            CUexecAffinityParam_v1 app_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)app_grid_size};
            CUexecAffinityParam_v1 mm_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)mm_grid_size};
            CUexecAffinityParam_v1 gc_param{CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, (unsigned int)gc_grid_size};

            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
            GUARD_CU((cudaError_t)cudaGetLastError());

            auto affinity_flags = CUctx_flags::CU_CTX_SCHED_AUTO;
            //auto affinity_flags = CUctx_flags::CU_CTX_SCHED_SPIN;
            //auto affinity_flags = CUctx_flags::CU_CTX_SCHED_YIELD;
            //auto affinity_flags = CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC;
            //auto affinity_flags = CUctx_flags::CU_CTX_BLOCKING_SYNC;
            //auto affinity_flags = CUctx_flags::CU_CTX_MAP_HOST;
            //auto affinity_flags = CUctx_flags::CU_CTX_LMEM_RESIZE_TO_MAX;

            CUcontext app_ctx, mm_ctx, gc_ctx;
   
            GUARD_CU((cudaError_t)cuCtxCreate_v3(&app_ctx,&app_param,1, affinity_flags, device));
            GUARD_CU((cudaError_t)cuCtxCreate_v3(&mm_ctx, &mm_param, 1, affinity_flags, device));
            GUARD_CU((cudaError_t)cuCtxCreate_v3(&gc_ctx, &gc_param, 1, affinity_flags, device));
            GUARD_CU(cudaPeekAtLastError());
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
            GUARD_CU((cudaError_t)cudaGetLastError());

            //Timing variables
            PerfMeasure malloc_total_sync, timing_mm, timing_gc;

            for (int iteration = 0; iteration < num_iterations; ++iteration){

                *exit_signal = 0;
                *exit_counter = 0;
                *mm_started = 0;
                *gc_started = 0;

                GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_signal, 
                                                    sizeof(int), device, NULL));
                GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)exit_counter, 
                                                    sizeof(int), device, NULL));
                GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)mm_started, 
                                                    sizeof(int), device, NULL));
                GUARD_CU((cudaError_t)cudaMemPrefetchAsync((int*)gc_started, 
                                                    sizeof(int), device, NULL));

                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU((cudaError_t)cudaGetLastError());

                RequestType requests;
                requests.init(requests_num);
                requests.memset();

                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());
                GUARD_CU((cudaError_t)cudaGetLastError());
                
                debug("start threads\n");

                // Run Memory Manager (Presistent kernel)
                std::thread mm_thread{[&] {
                    GUARD_CU((cudaError_t)cuCtxSetCurrent(mm_ctx));
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("start mm\n");
                    start_memory_manager(timing_mm, 
                            mm_numBlocksPerSm*mm_grid_size, 
                            block_size, mm_ctx, exit_signal, 
                            mm_started, requests, memory_manager);
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
                            block_size, gc_ctx, exit_signal, 
                            gc_started, requests, memory_manager);
                    debug("gc done, sync\n");
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    debug("done\n");
                }}; 

                //std::this_thread::sleep_for(std::chrono::seconds(1));

                //printf("-");
                //fflush(stdout);
                while (!(*gc_started && *mm_started));
                GUARD_CU((cudaError_t)cudaGetLastError());

                if (! (*gc_started)){
                    printf("gc did not start\n");
                }else{
                    printf("gc has started\n");
                }
                if (! (*mm_started)){
                    printf("mm did not start\n");
                }else{
                    printf("mm has started\n");
                }

                fflush(stdout);

                printf("app_numBlocksPerSm %d, app_grid_size %d, block_size %d\n", app_numBlocksPerSm, app_grid_size,
                block_size);

                // Run APP (all threads do malloc)
                bool kernel_complete = false;
                std::thread app_thread{[&] {
                    GUARD_CU((cudaError_t)cuCtxSetCurrent(app_ctx));
                    GUARD_CU((cudaError_t)cuCtxSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    GUARD_CU((cudaError_t)cudaGetLastError());
                    debug("start app\n");
                    start_application(MALLOC, malloc_total_sync, 
                            app_numBlocksPerSm*app_grid_size, 
                            block_size, app_ctx, exit_signal, 
                            requests, exit_counter, dev_size_to_alloc, 
                            dev_kernel_iteration_num, mono, 
                            kernel_complete, memory_manager);
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
                    clean_memory(app_grid_size, block_size, requests, 
                                 memory_manager, exit_signal);
                    continue;
                }

                *exit_signal = 1;

                debug("join mm\n");
                mm_thread.join();
                debug("mm joined\n");

                debug("join gc\n");
                gc_thread.join();
                debug("gc joined\n");

                //Deallocate device memory
                //cuCtxSetCurrent(default_ctx);
                //GUARD_CU((cudaError_t)cuCtxSetCurrent(default_ctx));
                clean_memory(app_grid_size, block_size, requests, memory_manager, exit_signal);
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());
            }
            printf("\n");
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

            printf("#measurements %d, mean %.2lf, #total iters %lu\n", malloc_total_sync_res.num_, malloc_total_sync_res.mean_, total_iters);

            printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
                    app_grid_size, mm_grid_size, gc_grid_size, uni_req_per_sec[it]);

            ++it;
        }
    }
    *array_size = it;
}

void pmm_init(int mono, int kernel_iteration_num, int size_to_alloc, size_t* ins_size, 
        size_t num_iterations, int SMs, int* sm_app, int* sm_mm, int* sm_gc, 
        int* allocs, float* uni_req_per_sec, int* array_size){


    printf("size to alloc per thread %d, num iterations %lu, kernel iterations %d, instantsize %lu, mono %d\n", 
            size_to_alloc, num_iterations, kernel_iteration_num, *ins_size, mono);

    std::cout << "#requests\t" << "#sm app\t\t" << "#sm mm\t\t" << "#sm gc\t\t" << "#malloc and free per sec\n";

    if (mono == MPS_mono){
        printf("MPS_mono\n");

        mps_monolithic_app(mono, kernel_iteration_num, size_to_alloc, ins_size, num_iterations, 
                SMs, sm_app, sm_mm, sm_gc, allocs, uni_req_per_sec, array_size);
    }else if (mono == simple_mono){
        printf("simple mono\n");

        simple_monolithic_app(mono, kernel_iteration_num, size_to_alloc, ins_size, num_iterations, 
                SMs, sm_app, sm_mm, sm_gc, allocs, uni_req_per_sec, array_size);
    }else{
        printf("MPS services\n");

        mps_app(mono, kernel_iteration_num, size_to_alloc, ins_size, num_iterations, 
                SMs, sm_app, sm_mm, sm_gc, allocs, uni_req_per_sec, array_size);
    }
}

}
