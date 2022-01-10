
#include "pmm-utils.cuh"

extern "C"{

__global__
void mem_free_perf(volatile int** d_memory, 
              MemoryManagerType* mm,
              int requests_num, 
              int turn_on
        ){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= requests_num){
        return;
    }
    if (turn_on){
            if (d_memory[thid])
                mm->free((void*)d_memory[thid]);
    }
}

__global__
void simple_alloc(volatile int** d_memory, 
        volatile int* exit_counter,
        MemoryManagerType* mm, 
        int size_to_alloc,
        int turn_on
        ){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (turn_on){
        d_memory[thid] = reinterpret_cast<volatile int*>
            (mm->malloc(size_to_alloc));
    }
    atomicAdd((int*)&exit_counter[0], 1);
}

void perf_alloc(int size_to_alloc, size_t* ins_size, size_t num_iterations, 
            int SMs, float* app_sync, float* uni_req_num, int turn_on){

    auto instant_size = *ins_size;

#ifdef OUROBOROS__
    //Ouroboros initialization
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);
#else
#ifdef HALLOC__
    //Halloc initialization
    //size_t instantitation_size = 2048ULL * 1024ULL * 1024ULL;
    MemoryManagerType memory_manager(instant_size);
#endif
#endif
 
    int* exit_counter;
    GUARD_CU(cudaMallocManaged(&exit_counter, sizeof(uint32_t)));
    GUARD_CU(cudaPeekAtLastError());

    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t app_stream;
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    
    int block_size = 1024;
    printf("size to alloc per thread %d, num iterations %d\n", size_to_alloc, num_iterations);
    std::cout << "\t\t#allocs\t\t" << "#sm app\t\t" << "#req per sec\t\t" << "app finish sync\n";

    for (int app_grid_size = 1; app_grid_size < SMs; ++app_grid_size){

        int requests_num{app_grid_size*block_size};
        //Timing variables
        PerfMeasure timing_total_sync;
        
        for (int iteration = 0; iteration < num_iterations; ++iteration){
            *exit_counter = 0;

            volatile int** d_memory{nullptr};
            GUARD_CU(cudaMalloc(&d_memory, requests_num * sizeof(volatile int*)));
            GUARD_CU(cudaPeekAtLastError());

            timing_total_sync.startMeasurement();
            //Run application
            simple_alloc<<<app_grid_size, block_size, 0, app_stream>>>(
                    d_memory,
                    exit_counter, 
#ifdef OUROBOROS__
                    memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
                    &memory_manager,
#endif
#endif                   
                    size_to_alloc, 
                    turn_on);
            GUARD_CU(cudaPeekAtLastError());

            // Check results
            int old_counter = -1;
            long long iter = 0;
            long long iter2 = 0;
            long long iter_mean = 0;
            long long  time_limit = 10000000000;
            while (iter2 < time_limit){
                if (exit_counter[0] == requests_num){
                    GUARD_CU(cudaDeviceSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    timing_total_sync.stopMeasurement();
                    if (turn_on){
                        test1<<<app_grid_size, block_size, 0, app_stream>>>(d_memory, requests_num);
                        GUARD_CU(cudaDeviceSynchronize());
                        GUARD_CU(cudaPeekAtLastError());
                        mem_test((int**)d_memory, requests_num, app_grid_size, block_size);
                    }
                    break;
                }else{
                    if (exit_counter[0] != old_counter){
                        old_counter = exit_counter[0];
                        ++iter;
                        iter_mean += iter2;
                        iter2 = 0;
                    }
                    ++iter2;
                }
                //was no change
                if (iter2 >= time_limit){
                    printf("time limit exceed, break\n");
                }
            }

            if (iter != 0)
                iter_mean /= iter;
        
            if (iter2 >= time_limit){
                printf("%d: sync\n", __LINE__);
            }
            GUARD_CU(cudaDeviceSynchronize());
            if (iter2 >= time_limit){
                printf("%d: sync done\n", __LINE__);
            }
            GUARD_CU(cudaPeekAtLastError());
            //Deallocate device memory
            mem_free_perf<<<app_grid_size, block_size, 0, app_stream>>>(
                    d_memory, 
#ifdef OUROBOROS__
                    memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
                    memory_manager,
#endif
#endif
                    requests_num, turn_on);
            GUARD_CU(cudaFree((void*)d_memory));
            GUARD_CU(cudaPeekAtLastError());
        }
        // Output
        auto total_sync_time = timing_total_sync.generateResult();
        app_sync  [app_grid_size - 1] = (total_sync_time.mean_);
        // The number of requests done per a second
        uni_req_num[app_grid_size - 1] = 
                            (requests_num * 1000.0)/total_sync_time.mean_;

        printf("\t\t%d\t\t| %d\t\t| %.2lf\t\t| %.2lf\n", 
                requests_num, app_grid_size, uni_req_num[app_grid_size - 1], 
                total_sync_time.mean_);
    }

    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
}


}
