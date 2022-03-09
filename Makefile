EXEC = main

$(EXEC):
 
	nvcc -G -g -O0 -Xptxas -O0 -arch=sm_75 --resource-usage -I include -lcuda main.cu -o simplest
    
	nvcc -G -g -O0 -Xptxas -O0 -arch=sm_75 --resource-usage --maxrregcount 24 -Xptxas -warn-lmem-usage -Xptxas \
    --warn-on-spills -I include -lcuda --compiler-options '-fPIC' -Xcompiler --shared pmm.cu -o simplest.so
 
 
  
ptx:

	nvcc -ptx -g -O0 -Xptxas -O3 -arch=sm_75 -I include -lcuda -o mock.ptx main.cu 

