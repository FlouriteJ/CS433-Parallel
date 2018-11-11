#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>


__constant__ int N;
__constant__ float a,b,c;



__global__ void cal(float *z, float *v)
{
	int ind;
	float sum = 0;
	float zi;
	// move array pointers to correct position
	ind = threadIdx.x + N*blockIdx.x*blockDim.x;
	for (int n=0; n<N; n++) {
		zi	 = z[ind];
		ind += blockDim.x;			// shift pointer to next element
		sum += (a * zi * zi + b * zi +c);
	}

	// put payoff value into device array

	v[threadIdx.x + blockIdx.x*blockDim.x] = sum/100;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
		
	int NCAL = 960000;
	int constN = 100;
	float *z, *v;

	// initialise card

	findCudaDevice(argc, argv);
	
	// initialise CUDA timing
	
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on host and device

	checkCudaErrors(cudaMallocManaged(&z, sizeof(float)*NCAL*constN));
	checkCudaErrors(cudaMallocManaged(&v, sizeof(float)*NCAL));
	
	/* 	h_v = (float *)malloc(sizeof(float)*NPATH);
	checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
	checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*NPATH) ); */
	

	// define constants and transfer to GPU

	float constA = 3.0f;
	float constB = 2.0f;
	float constC = 1.0f;
	
	printf("Initialized value: a = %.2f b = %.2f c = %.2f \n",constA, constB, constC);

	checkCudaErrors(cudaMemcpyToSymbol(N,&constN,sizeof(constN)));
	checkCudaErrors(cudaMemcpyToSymbol(a,&constA,sizeof(constA)));
	checkCudaErrors(cudaMemcpyToSymbol(b,&constB,sizeof(constB)));
	checkCudaErrors(cudaMemcpyToSymbol(c,&constC,sizeof(constC)));

	// random number generation

	cudaEventRecord(start);
	
	curandGenerator_t gen;
	checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	checkCudaErrors(curandGenerateNormal(gen, z, NCAL*constN, 0.0f, 1.0f));
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // ensure all the threads in GPU finish
	cudaEventElapsedTime(&milli, start, stop);

	printf("CURAND normal RNG	execution time (ms): %f,	samples/sec: %e \n",milli, NCAL*constN/(0.001*milli));

	// execute kernel and time it

	cudaEventRecord(start);

	cal<<<NCAL/64, 64>>>(z, v);
	getLastCudaError("cal execution failed\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);

	printf("Kernel execution time (ms): %f \n",milli);

	// synchronize to wait for kernel to finish, and data copied back

	cudaDeviceSynchronize();

	// compute average

	float result = 0.0;
	for (int i=0; i<NCAL; i++) {
		result += v[i];
	}

	printf("Average value = %13.8f \n", result/NCAL);

	// Tidy up library

	checkCudaErrors(curandDestroyGenerator(gen));

	// Release memory and exit cleanly

	checkCudaErrors(cudaFree(v));
	checkCudaErrors(cudaFree(z));
	
	/* 	free(h_v);
	checkCudaErrors( cudaFree(d_v) );
	checkCudaErrors( cudaFree(d_z) ); */

	// CUDA exit -- needed to flush printf write buffer

	cudaDeviceReset();

	return 0;
}
