
INC	:= -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -arch=sm_35 --ptxas-options=-v --use_fast_math

recequ:	recequ.cu
	nvcc recequ.cu -o recequ \
	$(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm recequ
