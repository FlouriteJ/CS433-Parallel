#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>

#define random() ((double(rand())/RAND_MAX) - 0.5)*2 // between 

#define epsilon 0.97

#define debug 1

void cpu_recque(double* s, double* u, int N, double* result){
	double s_sum = 1;
	double u_sum = 0;
	for(int i = 0;i<N;i++){
		u_sum+=(s_sum * u[i]);
		s_sum*=s[i];
		if(debug >= 2){
			printf("%.3lf %.3lf\n",u_sum,s_sum);
		}
	}
	result[0] = u_sum/(1-s_sum);
	for (int i = 1;i<N;i++){
		result[i] = (result[i-1] - u[i-1])/s[i-1];
	}
}

__global__ void gpu_recque(double* s, double* u, int N ,int nThreads, double* d_s_sum, double* d_u_sum){
	double s_sum = 1;
	double u_sum = 0;
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	int begin = floor(double(N)/nThreads) * tid;
	int end = floor(double(N)/nThreads) * (tid + 1);
	if(tid == nThreads - 1) end = N;
	
	for(int i = begin;i<end;i++){
		u_sum+=(s_sum * u[i]);
		s_sum*=s[i];
		if(debug == 3){
			printf("tid = %d  pos = %d\n",tid,i);
			printf("u = %.3lf s = %.3lf\n",u[i],s[i]);
			printf("%.3lf %.3lf\n",s_sum,u_sum);
		}
	}
	d_s_sum[tid] = s_sum;
	d_u_sum[tid] = u_sum;
}

__global__ void gpu_blocks_recque(double* s, double* u, int N ,int nThreads, double* d_s_sum, double* d_u_sum){
	extern  __shared__  double sum[]; // s_sum u_sum s_sum ... 
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	//printf("pid = %d\n",tid);
	if (tid >= N){
		sum[threadIdx.x*2] = 1;
		sum[threadIdx.x*2+1] = 0;
	}
	else{
		sum[threadIdx.x*2] = s[tid];
		sum[threadIdx.x*2+1] = u[tid];
	}
	
	if(debug == 2){
		printf("blockid = %d  pos = %d\n",blockIdx.x,tid);
		printf("u = %.3lf s = %.3lf\n",u[tid],s[tid]);
	}	
	
    for (int d = 1; d <= blockDim.x>>1; d <<= 1) {
		__syncthreads();
		if (threadIdx.x%(2*d)==0){
			sum[threadIdx.x*2+1]+=(sum[threadIdx.x*2] * sum[(threadIdx.x+d)*2+1]); // u_sum+=(s_sum * u[i])
			sum[threadIdx.x*2]*=sum[(threadIdx.x+d)*2]; // s_sum*=s[i];
			if(debug == 2){
				printf("tid = %d merging %d and %d   result: u = %.3lf s = %.3lf\n",tid,threadIdx.x,threadIdx.x+d,sum[threadIdx.x*2+1],sum[threadIdx.x*2]);
			}
		}  
    }	
	
	if (threadIdx.x==0){
		d_s_sum[blockIdx.x] = sum[0];
		d_u_sum[blockIdx.x] = sum[1];
	} 

}

double validation(double* s, double* u, int N, double* result){
	double sqrt_root_sum = 0;
	for (int i = 1;i<=N;i++){
		double t = fabs(s[i-1] * result[i==N?0:i] + u[i-1] - result[i-1]);
		sqrt_root_sum += (t*t);
	}
	return sqrt(sqrt_root_sum);
}
int main(int argc, const char **argv){
	srand(0);
	int N = 1000;
	double* s;
	double* u;
	double* result;		
	s = (double* )malloc(N*sizeof(double));
	u = (double* )malloc(N*sizeof(double));
	result = (double* )malloc(N*sizeof(double));
	for(int i = 0;i<N;i++){
		s[i] = 0;
		while(fabs(s[i] < epsilon)) s[i] = random();
		u[i] = 0;
		while(fabs(u[i] < epsilon)) u[i] = random();
	}
	if(debug){
		printf("The equation is: \n");
		for(int i = 1;i<=3;i++){
			printf("V[%d] = %.3lf*V[%d] + %.3lf \n",i-1,s[i-1],i==N?0:i,u[i-1]);
		}
		printf("...\n");	
		for(int i = N-2;i<=N;i++){
			printf("V[%d] = %.3lf*V[%d] + %.3lf \n",i-1,s[i-1],i==N?0:i,u[i-1]);			
		}
	}
	
	// CPU version
	cpu_recque(s,u,N,result);

	if(debug){
		printf("\nCPU result is :\n");
		for(int i = 0;i<3;i++){
			printf("V[%d] = %.3f \n",i,result[i]);
		}
		printf("...\n");	
		for(int i = N-3;i<N;i++){
			printf("V[%d] = %.3f \n",i,result[i]);	
		}			
	}
	
	double SRS = validation(s,u,N,result);
	printf("\nCPU Validation: \nSRS = %.3lf \n",SRS);
	
	for(int i = 0;i<N;i++){
		result[i] = 0;
	}
		
	// GPU version 1
	findCudaDevice(argc, argv);
	int nThreads = 16;
	double* d_s;
	double* d_u;
	double* d_s_sum;
	double* h_s_sum;
	double* d_u_sum;
	double* h_u_sum;	
	
	h_s_sum = (double *)malloc(nThreads*sizeof(double));
	h_u_sum = (double *)malloc(nThreads*sizeof(double));
	
	cudaMalloc((void **)&d_s, N*sizeof(double));
	cudaMalloc((void **)&d_u, N*sizeof(double));
	cudaMalloc((void **)&d_s_sum, nThreads*sizeof(double));
	cudaMalloc((void **)&d_u_sum, nThreads*sizeof(double));

	cudaMemcpy(d_s,s,N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_u,u,N*sizeof(double),cudaMemcpyHostToDevice);
	
	gpu_recque<<<1,nThreads>>>(d_s,d_u,N,nThreads,d_s_sum,d_u_sum);
	getLastCudaError("gpu_recque execution failed\n");
	
	checkCudaErrors(cudaMemcpy(h_u_sum,d_u_sum,nThreads*sizeof(double),cudaMemcpyDeviceToHost));	
	checkCudaErrors(cudaMemcpy(h_s_sum,d_s_sum,nThreads*sizeof(double),cudaMemcpyDeviceToHost));
	
	double u_sum = 0;
	double s_sum = 1;
	
	for(int i = 0;i<nThreads;i++){
		u_sum+=(s_sum * h_u_sum[i]);
		s_sum*=h_s_sum[i];		
	}
	
	result[0] = u_sum/(1-s_sum);
	
	for (int i = 1;i<N;i++){
		result[i] = (result[i-1] - u[i-1])/s[i-1];
	}
	
	if(debug){
		printf("\nGPU result is :\n");
		for(int i = 0;i<3;i++){
			printf("V[%d] = %.3f \n",i,result[i]);
		}
		printf("...\n");	
		for(int i = N-3;i<N;i++){
			printf("V[%d] = %.3f \n",i,result[i]);	
		}			
	}
	
	SRS = validation(s,u,N,result);
	printf("\nGPU Validation: \nSRS = %.3lf \n",SRS);
	
	
	
	// GPU version 2: MultiBlocks Version
	for(int i = 0;i<N;i++){
		result[i] = 0;
	}
	checkCudaErrors(cudaFree(d_s_sum));
	checkCudaErrors(cudaFree(d_u_sum));
	free(h_s_sum);
	free(h_u_sum);
	
	nThreads = 16;
	int nBlocks = ceil(double(N)/nThreads);
	int shared_mem_size = nBlocks*2*sizeof(double);
	
	h_s_sum = (double *)malloc(nBlocks*sizeof(double));
	h_u_sum = (double *)malloc(nBlocks*sizeof(double));
	
	cudaMalloc((void **)&d_s_sum, nBlocks*sizeof(double));
	cudaMalloc((void **)&d_u_sum, nBlocks*sizeof(double));
	
	gpu_blocks_recque<<<nBlocks,nThreads,shared_mem_size>>>(d_s,d_u,N,nThreads,d_s_sum,d_u_sum);
	getLastCudaError("gpu_blocks_recque execution failed\n");
	
	checkCudaErrors(cudaMemcpy(h_u_sum,d_u_sum,nBlocks*sizeof(double),cudaMemcpyDeviceToHost));	
	checkCudaErrors(cudaMemcpy(h_s_sum,d_s_sum,nBlocks*sizeof(double),cudaMemcpyDeviceToHost));
	
	u_sum = 0;
	s_sum = 1;
	
	for(int i = 0;i<nBlocks;i++){
		u_sum+=(s_sum * h_u_sum[i]);
		s_sum*=h_s_sum[i];		
	}
	
	result[0] = u_sum/(1-s_sum);
	
	for (int i = 1;i<N;i++){
		result[i] = (result[i-1] - u[i-1])/s[i-1];
	}
	
	if(debug){
		printf("\nGPU(multiblocks) result is :\n");
		for(int i = 0;i<3;i++){
			printf("V[%d] = %.3f \n",i,result[i]);
		}
		printf("...\n");	
		for(int i = N-3;i<N;i++){
			printf("V[%d] = %.3f \n",i,result[i]);	
		}			
	}
	
	SRS = validation(s,u,N,result);
	printf("\nGPU(multiblocks) Validation: \nSRS = %.3lf \n",SRS);	
}