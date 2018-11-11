//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

int BLOCK_X, BLOCK_Y, BLOCK_Z;
////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <laplace3d_kernel_new_optimize.h>


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  // 'h_' prefix - CPU (host) memory space

  int    NX=256, NY=256, NZ=256, REPEAT=50,
         bx, by, bz, i, j, k, ind;
  float  *h_u1, *h_u2, *h_u3, *h_foo, err;

  // 'd_' prefix - GPU (device) memory space

  float  *d_u1, *d_u2, *d_foo;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
 for (BLOCK_X = 1;BLOCK_X<=256;BLOCK_X*=2){
	  for (BLOCK_Y = 1;BLOCK_Y<=256;BLOCK_Y*=2){
		   for (BLOCK_Z = 1;BLOCK_Z<=64;BLOCK_Z*=2){
			  // allocate memory for arrays
			  if (BLOCK_X * BLOCK_Y * BLOCK_Z > 1024) break;

			  h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
			  h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
			  h_u3 = (float *)malloc(sizeof(float)*NX*NY*NZ);
			  checkCudaErrors( cudaMalloc((void **)&d_u1, sizeof(float)*NX*NY*NZ) );
			  checkCudaErrors( cudaMalloc((void **)&d_u2, sizeof(float)*NX*NY*NZ) );

			  // initialise u1

			  for (k=0; k<NZ; k++) {
				for (j=0; j<NY; j++) {
				  for (i=0; i<NX; i++) {
					ind = i + j*NX + k*NX*NY;

					if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
					  h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
					else
					  h_u1[ind] = 0.0f;
				  }
				}
			  }

			  // copy u1 to device

			  checkCudaErrors( cudaMemcpy(d_u1, h_u1, sizeof(float)*NX*NY*NZ,
										  cudaMemcpyHostToDevice) );

			  // Set up the execution configuration

			  
			  bx = ceil((NX-1.0)/BLOCK_X);
			  by = ceil((NY-1.0)/BLOCK_Y);
			  bz = ceil((NZ-1.0)/BLOCK_Z);

			  dim3 dimGrid(bx,by,bz);
			  dim3 dimBlock(BLOCK_X,BLOCK_Y,BLOCK_Z);

			  // printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
			  // printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

			  // Execute GPU kernel

			  cudaEventRecord(start);

			  for (i = 1; i <= REPEAT; ++i) {
				GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2, BLOCK_X, BLOCK_Y, BLOCK_Z);
				getLastCudaError("GPU_laplace3d execution failed\n");

				d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;   // swap d_u1 and d_u2
			  }

			  cudaEventRecord(stop);
			  cudaEventSynchronize(stop);
			  cudaEventElapsedTime(&milli, start, stop);
			  printf("%d %d %d %.2f \n", BLOCK_X, BLOCK_Y, BLOCK_Z, milli);

			  // Read back GPU results

			  checkCudaErrors( cudaMemcpy(h_u2, d_u1, sizeof(float)*NX*NY*NZ,
										  cudaMemcpyDeviceToHost) );


			 // Release GPU and CPU memory

			  checkCudaErrors( cudaFree(d_u1) );
			  checkCudaErrors( cudaFree(d_u2) );
			  free(h_u1);
			  free(h_u2);
			  free(h_u3);
		   }
	  }
 }
  cudaDeviceReset();
}
