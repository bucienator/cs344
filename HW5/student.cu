/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

__global__
void histoClear(unsigned int * const histo,
				const unsigned int numBins)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < numBins) {
	  histo[idx] = 0;
  }
}

__global__
void yourHisto(const unsigned int* const vals, //INPUT
			   unsigned int* const histo,      //OUPUT
			   const unsigned int numBins,
			   const unsigned int numElems)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < numElems) {
	  unsigned int value = vals[idx];
	  atomicAdd(&histo[value], 1);
  }

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
					  unsigned int* const d_histo,      //OUTPUT
					  const unsigned int numBins,
					  const unsigned int numElems)
{
	{
		dim3 blockDim(1024);
		dim3 gridDim(numBins/blockDim.x + (numBins % blockDim.x == 0 ? 0 : 1) );

		histoClear<<<gridDim, blockDim>>>(d_histo, numBins);
	}

	{
		dim3 blockDim(1024);
		dim3 gridDim(numElems/blockDim.x + (numElems % blockDim.x == 0 ? 0 : 1) );

		yourHisto<<<gridDim, blockDim>>>(d_vals, d_histo, numBins, numElems);
	}

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
