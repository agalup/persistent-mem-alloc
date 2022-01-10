
cudaError_t GRError(cudaError_t error, const char *message,
                    const char *filename, int line, bool print) {
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename,
            line, gpu, message, error, cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

#define GUARD_CU(cuda_call)                                                   \
  {                                                                           \
    if (cuda_call != (enum cudaError) CUDA_SUCCESS){  \
        printf("--- ERROR(%d:%s) --- %s:%d\n", cuda_call, cudaGetErrorString(cuda_call), __FILE__, __LINE__);\
    } \
  }\

struct Memory_List{
};

struct MemoryAllocator{

    int size;
    int stack_top;
    void* memory;
    void* 
    volatile int* requests_number; 
    volatile int* request_iter;
    volatile int* request_signal; 
    volatile int* request_id; 
    volatile int* request_mem_size;
    volatile int* lock;
    volatile int** d_memory{nullptr};

    void init(size_t Size);
    void memset();
    void free();
    void malloc();
};

void MemoryAllocator::init(size_t Size){
    size = Size;
    
    GUARD_CU(cudaMallocManaged(&memory, size * sizeof(volatile void)));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
}


