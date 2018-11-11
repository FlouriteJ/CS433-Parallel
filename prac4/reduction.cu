#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

void reduction_gold(float* odata, float* idata, const unsigned int len) 
{
  *odata = 0;
  for(int i=0; i<len; i++) *odata += idata[i];
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
	
	// version 1:shared memory
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // first, each thread loads data into shared memory

    temp[threadIdx.x] = g_idata[tid];

    // next, we perform binary tree reduction

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();  // ensure previous step completed 
      if (threadIdx.x<d)  temp[threadIdx.x] += temp[tid+d];
    }

    // finally, first thread puts result into global memory

    if (threadIdx.x==0) g_odata[blockIdx.x] = temp[0];
	
	//version 2: shuffle
/*
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	float sum = g_idata[tid];

    // next, we perform binary tree reduction

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();  // ensure previous step completed 
      sum+=__shfl_down(sum,d);
    }

    // finally, first thread puts result into global memory

    if (threadIdx.x==0) g_odata[blockIdx.x] = sum; */
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_elements, num_threads, mem_size, shared_mem_size;

  float *h_data, *reference, sum;
  float *d_idata, *d_odata;
  
  int nBlock = 16;
  
  // initialise card

  findCudaDevice(argc, argv);

  num_elements = 512;
  num_threads  = num_elements;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

  // compute reference solutions

  reference = (float*) malloc(mem_size);
  reduction_gold(&sum, h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float) * nBlock) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_elements;
  reduction<<<nBlock,num_threads/nBlock,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, sizeof(float)*nBlock,
                              cudaMemcpyDeviceToHost) );

  // check results

  int result = 0;
  for (int i = 0;i<nBlock;i++){
	  result+=h_data[i];
  }
  printf("reduction error = %f\n",result-sum);

  // cleanup memory

  free(h_data);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
