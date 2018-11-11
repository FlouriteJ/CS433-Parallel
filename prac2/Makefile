
INC	:= -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo  --ptxas-options=-v --use_fast_math

all: prac2 prac2me

prac2: prac2.cu Makefile
	nvcc prac2.cu -o prac2 $(INC) $(NVCCFLAGS) $(LIB)

prac2me: prac2me.cu Makefile
	nvcc prac2me.cu -o prac2me $(INC) $(NVCCFLAGS) $(LIB)
	
clean:
	rm -f prac2  prac2me