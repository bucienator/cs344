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
	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numBins; i += gridDim.x * blockDim.x) {
		histo[i] = 0;
	}
}

__global__
void histoBlock(const unsigned int * const d_input,
				unsigned int * const d_histo,
				const size_t numElements,
				const size_t numBins)
{
	extern __shared__ unsigned int histo[];

	for(size_t i = threadIdx.x; i < numBins; i+=blockDim.x) {
		histo[i] = 0;
	}

	__syncthreads();

	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements; i+=gridDim.x * blockDim.x) {
		atomicAdd(&histo[d_input[i]], 1);
	}

	__syncthreads();

	for(size_t i = threadIdx.x; i < numBins; i+=blockDim.x) {
		atomicAdd(&d_histo[i], histo[i]);
	}

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
					  unsigned int* const d_histo,      //OUTPUT
					  const unsigned int numBins,
					  const unsigned int numElems)
{
	histoClear<<<1,1024>>>(d_histo, numBins);
	histoBlock<<<1024,512, numBins*sizeof(unsigned int)>>>(d_vals, d_histo, numElems, numBins);
	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
