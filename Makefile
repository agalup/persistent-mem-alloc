EXEC = main
$(EXEC):
	mkdir -p cubin

#OUROBOROS
	#nvcc -cubin -O0 -Xptxas -O0 -arch=sm_75 \
    #--expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ main.cu \
    #-lcuda -I SlabHash -I SlabHash/SlabAlloc/src -o cubin/ouroboros_mm_O0.cubin
	#
	#nvcc -cubin -O1 -Xptxas -O1 -arch=sm_75 \
    #--expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ main.cu \
    #-lcuda -I SlabHash -I SlabHash/SlabAlloc/src -o cubin/ouroboros_mm_O1.cubin
	#
	#nvcc -cubin -O2 -Xptxas -O2 -arch=sm_75 \
    #--expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ main.cu \
    #-lcuda -I SlabHash -I SlabHash/SlabAlloc/src -o cubin/ouroboros_mm_O2.cubin

	#nvcc -cubin -O3 -Xptxas -O3 -arch=sm_75 \
    #--expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ main.cu \
    #-lcuda -I SlabHash -I SlabHash/SlabAlloc/src -o cubin/ouroboros_mm_O3.cubin
	
#	nvcc -O3 -Xptxas -O3 -arch=sm_70 --resource-usage -lineinfo -Xptxas --warn-on-spills \
#    --expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ main.cu \
#    -lcuda -I SlabHash -I SlabHash/SlabAlloc/src -o ouroboros_mm
#	
#	nvcc -g -O3 -Xptxas -O3 -arch=sm_70 --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 \
#    --expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ \
#    -lcuda --compiler-options '-fPIC' -Xcompiler --shared pmm.cu -o ouroboros_mm.so 

	#nvcc -g -G -O0 -Xptxas -O0 -arch=sm_70 \
    #--compiler-options '-fPIC' -Xcompiler  -lcuda -lcudart -I include -I Ouroboros_origin/include -DOUROBOROS__ main.cu -o ouroboros_mm --expt-relaxed-constexpr

	nvcc -g -O3 -Xptxas -O3 -arch=sm_70 --maxrregcount 32 \
    -lcuda -lcudart -I include -I Ouroboros_origin/include \
    -DOUROBOROS__ main.cu --expt-relaxed-constexpr --expt-extended-lambda -o ouroboros_mm 
	
	nvcc -g -O3 -Xptxas -O3 -arch=sm_70 --maxrregcount 32 \
    -lcuda -lcudart -I include -I Ouroboros_origin/include \
    -DOUROBOROS__  --expt-relaxed-constexpr --expt-extended-lambda\
    --compiler-options '-fPIC' --shared pmm.cu -o ouroboros_mm.so

#	nvcc -g -G -O0 -Xptxas -O0 -arch=sm_70 --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 \
#    --expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ \
#    -lcuda --compiler-options '-fPIC' -Xcompiler --shared pmm.cu -o ouroboros_mm.so 

