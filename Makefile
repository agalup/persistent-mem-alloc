EXEC = main

$(EXEC):
 
	#nvcc -G -g -v -arch=sm_75 --resource-usage -Xptxas -warn-lmem-usage -Xptxas --warn-on-spills 
    #-res-usage --maxrregcount 32 --expt-relaxed-constexpr -I include -lcuda main.cu -o mock

	nvcc -O3 -Xptxas -O3 -arch=sm_70 --resource-usage -I include -lcuda main.cu -o mock
	#nvcc -O3 -Xptxas -O3 -arch=sm_75 --resource-usage -I include -lcuda main.cu -o mock
    
	nvcc -O3 -Xptxas -O3 -arch=sm_70 --resource-usage --maxrregcount 24 -Xptxas -warn-lmem-usage -Xptxas --warn-on-spills -I include -lcuda \
	--compiler-options '-fPIC' -Xcompiler --shared pmm.cu -o mock.so
 
#	nvcc -G -g -arch=sm_75 --resource-usage -Xptxas -warn-lmem-usage -Xptxas --warn-on-spills \
#    -res-usage --maxrregcount 32 --expt-relaxed-constexpr -I include -lcuda \
#    --compiler-options '-fPIC' -Xcompiler --shared pmm.cu -o mock.so
 
  
ptx:

	nvcc -ptx -g -O0 -Xptxas -O3 -arch=sm_75 -I include -lcuda -o mock.ptx main.cu 

