EXEC = main

$(EXEC):
 
#OUROBOROS
	nvcc -G -g -arch=sm_70 --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 \
    --expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ main.cu \
    -lcuda -I SlabHash -I SlabHash/SlabAlloc/src -o ouroboros_mm
	
	nvcc -G -g -arch=sm_70 --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 \
    --expt-relaxed-constexpr -I include -I Ouroboros_origin/include -DOUROBOROS__ \
    -lcuda -I SlabHash -I SlabHash/SlabAlloc/src \
    --compiler-options '-fPIC' -Xcompiler --shared pmm.cu -o ouroboros_mm.so 
 
## HALLOC:
#	nvcc -G -g --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 \
#    --expt-relaxed-constexpr -L cuda.h -I GPUMemManSurvey/include \
#    -I GPUMemManSurvey/frameworks/halloc/repository/src/ \
#    -I GPUMemManSurvey/frameworks/ouroboros/repository/include -I include \
#    -I GPUMemManSurvey/frameworks/halloc -DHALLOC__ main.cu -o halloc_mm
#
#	nvcc -G -g --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 \
#    --expt-relaxed-constexpr -L cuda.h -I GPUMemManSurvey/include \
#    -I GPUMemManSurvey/frameworks/halloc/repository/src/ \
#    -I GPUMemManSurvey/frameworks/ouroboros/repository/include -I include \
#    -I GPUMemManSurvey/frameworks/halloc -DHALLOC__ \
#    --compiler-options '-fPIC' -Xcompiler --shared pmm.cu -o halloc_mm.so


    
