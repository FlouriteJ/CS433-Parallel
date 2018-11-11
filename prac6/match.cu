//
// Pattern-matching program
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

__global__ void gpu_match(unsigned int *text, unsigned int *words, int *h_matches, int nwords, int length, int nThreads)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  int texti, texti2, word;
  
  // printf("tid=%d \n",tid);
  
  while (tid < length){  
	  texti = text[tid];
	  texti2 = text[tid+1];
	  
	  for (int offset=0; offset<4; offset++) {
		if (offset==0)
		  word = texti;
		else
		  word = (texti>>(8*offset)) + (texti2<<(32-8*offset)); 
		for (int w=0; w<nwords; w++) {		  
		  if (word==words[w]) {
			h_matches[w + nwords*(threadIdx.x + blockDim.x*blockIdx.x)] +=1;
			// printf("match: %d \n",h_matches[w + nwords*threadIdx.x]);
			// matches[w] += 1;
			// printf("match: tid=%d, word=%d \n",tid,word);
		  }
		}
		
	  }
	  tid += nThreads;
  }
}


////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void gold_match(unsigned int *, unsigned int *, int *, int, int);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
  char *ctext, *cwords[] = {"cuti", "gold", "text", "word"};
  unsigned int  *text,  *words;

  int   length, len, nwords=4, matches[4]={0, 0, 0, 0};

  // read in text for processing

  FILE *fp;
  fp = fopen("match.cu","r");

  length = 0;
  while (getc(fp) != EOF) length++;

  ctext = (char *) malloc(length+4);

  rewind(fp);

  for (int l=0; l<length; l++) ctext[l] = getc(fp);
  for (int l=length; l<length+4; l++) ctext[l] = ' ';

  fclose(fp);

  // define number of words of text, and set pointers

  len  = length/4;
  text = (unsigned int *) ctext;

  // define words for matching

  words = (unsigned int *) malloc(nwords*sizeof(unsigned int));

  for (int w=0; w<nwords; w++) {
    words[w] = ((unsigned int) cwords[w][0])
             + ((unsigned int) cwords[w][1])*256
             + ((unsigned int) cwords[w][2])*256*256
             + ((unsigned int) cwords[w][3])*256*256*256;
  }

  // CPU execution

  gold_match(text, words, matches, nwords, len);

  printf(" CPU matches = %d %d %d %d \n",
         matches[0],matches[1],matches[2],matches[3]);

  for (int w=0; w<nwords; w++) matches[w] = 0;
  
  // GPU execution
  unsigned int  *d_text,  *d_words;
  int *d_matches, *h_matches;
  int nThreads = 16;
  
  findCudaDevice(argc, argv);
  checkCudaErrors(cudaMalloc((void **)&d_words, nwords*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **)&d_text, (length+nwords)*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **)&d_matches, nThreads*nwords*sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_words,words,nwords*sizeof(unsigned int),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_text,text,(length+nwords)*sizeof(unsigned int),cudaMemcpyHostToDevice));
  
  
  h_matches = (int *) malloc(nThreads*nwords*sizeof(int));
  for (int i = 0;i<nThreads*nwords;i++) h_matches[i] = 0;
  checkCudaErrors(cudaMemcpy(d_matches,h_matches,nThreads*nwords*sizeof(unsigned int),cudaMemcpyHostToDevice));
  
  
  
  
  gpu_match<<<1,nThreads>>>(d_text,d_words,d_matches,nwords,len,nThreads);
  getLastCudaError("gpu_match execution failed\n");
  
  
  
  checkCudaErrors(cudaMemcpy(h_matches,d_matches,nThreads*nwords*sizeof(unsigned int),cudaMemcpyDeviceToHost));
  
  
  for (int i = 0;i<nThreads*nwords;i++) {
	  matches[i%nwords] += h_matches[i];
	  // printf("%d \n",h_matches[i]); 
  }
  
  printf(" GPU matches = %d %d %d %d \n",
  matches[0],matches[1],matches[2],matches[3]);    
  // Release GPU and CPU memory
  
  checkCudaErrors(cudaFree(d_words));
  checkCudaErrors(cudaFree(d_matches));
  checkCudaErrors(cudaFree(d_text));
  free(ctext);
  
  cudaDeviceReset();
  return 0;
}
