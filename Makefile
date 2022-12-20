EXEC = main
$(EXEC):
	mkdir -p cubin

	nvcc -g -O3 -Xptxas -O3 -arch=sm_70 --maxrregcount 32 \
    -lcuda -lcudart -I include -I Ouroboros_origin/include \
    -DOUROBOROS__ main.cu --expt-relaxed-constexpr --expt-extended-lambda -o ouroboros_mm 
	
	nvcc -g -O3 -Xptxas -O3 -arch=sm_70 --maxrregcount 32 \
    -lcuda -lcudart -I include -I Ouroboros_origin/include \
    -DOUROBOROS__  --expt-relaxed-constexpr --expt-extended-lambda\
    --compiler-options '-fPIC' --shared pmm.cu -o ouroboros_mm.so

	nvcc -g -O3 -Xptxas -O3 -arch=sm_70 --maxrregcount 32 \
    -lcuda -lcudadevrt -I include -I Ouroboros_origin/include \
    main.cu --expt-relaxed-constexpr --expt-extended-lambda -o cudaMalloc_mm 
	
