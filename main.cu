#include "pmm.cu"
//#include "monolitic_mm.cu"
#include "pmm-utils.cuh"

using namespace std;

int main(int argc, char *argv[]){
    
    size_t instant_size = 16 * 1024ULL * 1024ULL * 1024ULL;
    //size_t instant_size = 8 * 1024ULL * 1024ULL * 1024ULL;
    //size_t instant_size = 7 * 1024ULL * 1024ULL * 1024ULL;
    int size_to_alloc = 32;
    int iteration_num = 1;
    int kernel_iter_num = 1;
    int mono = 0;
    int device_id = 0;
    int block_size = 1024;
    int cb_number = 100;
    
    if (argc > 1){
        size_to_alloc = atoi(argv[1]);
    }
    if (argc > 2){
        mono = atoi(argv[2]);
    }
    if (argc > 3){
        iteration_num = atoi(argv[3]);
    }
    if (argc > 4){
        block_size = atoi(argv[4]);
    }
    if (argc > 5){
        device_id = atoi(argv[5]);
    }
    if (argc > 6){
        kernel_iter_num = atoi(argv[6]);
    }
    if (argc > 7){
        cb_number = atoi(argv[7]);
    }
    
    printf("./a.out <B to alloc> <mono?> <#iters> <block_size> <device id> <kernel iters> <callback num>\n");
    printf("./a.out       %d        %d      %d          %d          %d          %d              %d\n", 
            size_to_alloc, mono, iteration_num, block_size, device_id, kernel_iter_num, cb_number);

    cudaDeviceProp deviceProp;
    GUARD_CU(cudaGetDeviceProperties(&deviceProp, 0));
    int SMs = deviceProp.multiProcessorCount;
    printf("max block number %d\n", SMs);
    printf("instant size %ld\n", instant_size);
    int size = (SMs-2)*(SMs-3)*(SMs-4);

    int* sm_app             = (int*)malloc(sizeof(int)*size);
    int* sm_mm              = (int*)malloc(sizeof(int)*size);
    int* sm_gc              = (int*)malloc(sizeof(int)*size);
    int* allocs_size        = (int*)malloc(sizeof(int)*size);
    float* uni_req_per_sec   = (float*)malloc(sizeof(float)*size);
    int* array_size         = new int(0);
    
    pmm_init(mono, kernel_iter_num, size_to_alloc, &instant_size, 
            iteration_num, SMs, sm_app, sm_mm, sm_gc, allocs_size, 
            uni_req_per_sec, array_size, block_size, device_id, cb_number);

    GUARD_CU(cudaDeviceReset());
    GUARD_CU(cudaPeekAtLastError());

    printf("DONE!\n");
    return 0;
}

