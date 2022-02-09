EXEC = main

$(EXEC):
 
	nvcc -G -g -v -arch=sm_75 --resource-usage -Xptxas -warn-lmem-usage -Xptxas --warn-on-spills \
    -res-usage --maxrregcount 32 --expt-relaxed-constexpr -I include -lcuda main.cu -o matrix_mul

	nvcc -G -g -v -arch=sm_75 --resource-usage -Xptxas -warn-lmem-usage -Xptxas --warn-on-spills \
    -res-usage --maxrregcount 32 --expt-relaxed-constexpr -I include -lcuda \
    --compiler-options '-fPIC' -Xcompiler --shared persistent_manager.cu -o matrix_mul.so
 
   
