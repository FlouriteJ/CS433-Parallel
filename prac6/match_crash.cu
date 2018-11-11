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

__global__ void gpu_match(unsigned int *text, unsigned int *words, int *matches, int nwords, int length)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid > length) return;
  
  int texti = text[tid];
  int texti2 = text[tid+1];
  
  int word;
  
  for (int offset=0; offset<4; offset++) {
    if (offset==0)
	  word = texti;
    else
	  word = (texti>>(8*offset)) + (texti2<<(32-8*offset)); 
    for (int w=0; w<nwords; w++) {
	  matches[w] += (word==words[w]);
	  
	  if (word==words[w]) {
        // printf("%s match: l=%d, word=%d \n",tid,word);
      }
    }
	
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
  printf("Begin \n\n")
  findCudaDevice(argc, argv);
  char *ctext, *cwords[] = {"cuti", "gold", "text", "word"};
  unsigned int  *text,  *words;

  int   length, len, nwords=4, *matches;
  
  int nThreads = 64;
  
  checkCudaErrors(cudaMallocManaged(&matches, nwords*sizeof(int)));
  
  for(int i = 0;i<nwords;i++){
	  matches[i] = 0;
  }

  
  // read in text for processing

  FILE *fp;
  fp = fopen("match.cu","r");

  length = 0;
  while (getc(fp) != EOF) length++;

  checkCudaErrors(cudaMallocManaged(&ctext, (length+nwords)*sizeof(unsigned int)));
  
  rewind(fp);

  for (int l=0; l<length; l++) ctext[l] = getc(fp);
  for (int l=length; l<length+4; l++) ctext[l] = ' ';

  fclose(fp);

  // define number of words of text, and set pointers

  len  = length/4;
  text = (unsigned int *) ctext;
  

  // define words for matching

  checkCudaErrors(cudaMallocManaged(&words, nwords*sizeof(unsigned int)));

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

  for(int i = 0;i<nwords;i++){
	  matches[i] = 0;
  }

  printf(" cleaned matches = %d %d %d %d \n",
  matches[0],matches[1],matches[2],matches[3]);  
  
  // GPU execution

  // checkCudaErrors(cudaMallocManaged(&nwords, sizeof(int)));
  // checkCudaErrors(cudaMallocManaged(&len, sizeof(int)));  
  
  gpu_match<<<1,nThreads>>>(text,words,matches,nwords,len);
  getLastCudaError("gpu_match execution failed\n");
  
  cudaDeviceSynchronize();
  
  printf(" GPU matches = %d %d %d %d \n",
  matches[0],matches[1],matches[2],matches[3]);
  
  
  // Release GPU and CPU memory

  
  checkCudaErrors(cudaFree(words));
  checkCudaErrors(cudaFree(matches));
  checkCudaErrors(cudaFree(text));
  free(ctext);
  
  cudaDeviceReset();
  return 0;
}
