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
//#include <thrust/host_vector.h>
#include <vector>
#include <algorithm>


__device__ inline unsigned int max2(unsigned int a, unsigned int b)
{
	return a > b ? a : b;
}

__device__ inline unsigned int min2(unsigned int a, unsigned int b)
{
	return a < b ? a : b;
}

__device__ inline unsigned int add2(unsigned int a, unsigned int b)
{
	return a + b;
}

template <unsigned int(*func)(unsigned int, unsigned int)>
__global__
void exclusive_scan_add_carry(unsigned int * const d_data, const size_t numElems, const unsigned int * const d_carry)
{
	const size_t fullIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int * const blockStart = d_data + blockIdx.x * blockDim.x;

	if(fullIdx < numElems)
		blockStart[threadIdx.x] = func(blockStart[threadIdx.x], d_carry[blockIdx.x]);
}


template <unsigned int(*func)(unsigned int, unsigned int), unsigned int zero>
__global__
void exclusive_scan_block(unsigned int * const d_data, const size_t numElems, unsigned int * const d_carry)
{
	extern __shared__ unsigned int s_block[];

	const size_t fullIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int * const blockStart = d_data + blockIdx.x * blockDim.x;

	if(fullIdx < numElems)
		s_block[threadIdx.x] = blockStart[threadIdx.x];
	else
		s_block[threadIdx.x] = zero;

	if(threadIdx.x == blockDim.x-1) {
		d_carry[blockIdx.x] = s_block[threadIdx.x];
	}

	__syncthreads();

	int step = 1;

	while(step < blockDim.x) {
		if(threadIdx.x % (step * 2) == 0 && (threadIdx.x + step) < blockDim.x) {
			s_block[blockDim.x - 1 - threadIdx.x] = func(s_block[blockDim.x - 1 - threadIdx.x], s_block[blockDim.x - 1 - threadIdx.x - step]);
		}
		step *= 2;

		__syncthreads();
	}

	s_block[blockDim.x - 1] = zero;
	__syncthreads();

	do {
		step /= 2;
		if(threadIdx.x % (step * 2) == 0 && (threadIdx.x + step) < blockDim.x) {
			unsigned int temp = s_block[blockDim.x - 1 - threadIdx.x];
			s_block[blockDim.x - 1 - threadIdx.x] = func(s_block[blockDim.x - 1 - threadIdx.x], s_block[blockDim.x - 1 - threadIdx.x - step]);
			s_block[blockDim.x - 1 - threadIdx.x - step] = temp;
		}
		__syncthreads();
	} while(step > 1);

	if(fullIdx < numElems)
		blockStart[threadIdx.x] = s_block[threadIdx.x];

	if(threadIdx.x == blockDim.x-1) {
		d_carry[blockIdx.x] = func(d_carry[blockIdx.x], s_block[threadIdx.x]);
	}

}


template <unsigned int(*func)(unsigned int, unsigned int), unsigned int zero>
unsigned int exclusive_scan(unsigned int * d_data, const size_t numElems)
{
	const dim3 blockSize (1024, 1, 1);
	const dim3 gridSize (numElems/blockSize.x + ((numElems%blockSize.x) == 0 ? 0 : 1), 1, 1);

	// allocate carry
	unsigned int * d_carry;
	checkCudaErrors(cudaMalloc(&d_carry, gridSize.x * sizeof(unsigned int)));

	// scan blocks
	exclusive_scan_block<func, zero> <<<gridSize, blockSize, blockSize.x*sizeof(unsigned int)>>>(d_data, numElems, d_carry);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );

	unsigned int ret;

	// recursive scan carry
	if(gridSize.x > 1) {
		ret = exclusive_scan<func, zero> (d_carry, gridSize.x);

		// add carry
		exclusive_scan_add_carry<func> <<<gridSize, blockSize>>>(d_data, numElems, d_carry);
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
	} else {
		checkCudaErrors(cudaMemcpy(&ret, d_carry, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	}


	// deallocate carry
	checkCudaErrors(cudaFree(d_carry));

	return ret;
}


__global__
void radix_sort_assign(const unsigned int* const d_inputVals,
			   unsigned int* const d_selectors,
			   const size_t numElems,
			   const unsigned int blockMask,
			   const unsigned int baseBit)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < numElems) {
		for(int i = 0; i < blockMask+1; i++) {
			d_selectors[i * numElems + idx] = 0;
		}
		unsigned int block = d_inputVals[idx];
		block = (block >> baseBit) & blockMask;
		d_selectors[block * numElems + idx] = 1;
	}
}


__global__
void radix_sort_scatter(const unsigned int* const d_inputVals,
					   unsigned int* const d_outputVals,
					   const size_t numElems,
					   const unsigned int * const d_selectors,
					   const unsigned int blockMask,
					   const unsigned int baseBit)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < numElems) {
		unsigned int block = d_inputVals[idx];
		block = (block >> baseBit) & blockMask;
		unsigned int pos = d_selectors[block * numElems + idx];

		d_outputVals[pos] = d_inputVals[idx];
	}

}

void your_sort(unsigned int* const d_inputVals,
			   unsigned int* const d_outputVals,
			   const size_t numElems)
{
	
	const unsigned int baseBitIncrement = 4;
	const unsigned int selectorMask = (1 << baseBitIncrement) - 1;

	unsigned int * d_selectors;
	checkCudaErrors(cudaMalloc(&d_selectors, (selectorMask+1) * numElems * sizeof(unsigned int)));
  
	const dim3 blockSizeAssign (1024, 1, 1);
	const dim3 gridSizeAssign (numElems/blockSizeAssign.x + ((numElems%blockSizeAssign.x) == 0 ? 0 : 1), 1, 1);

 
	unsigned int * values[] = {d_inputVals, d_outputVals};
	int target = 1;
	for(unsigned int baseBit = 0; baseBit < sizeof(unsigned int) * 8; baseBit += baseBitIncrement) {
		radix_sort_assign<<<gridSizeAssign,blockSizeAssign>>>(values[1-target], d_selectors, numElems, selectorMask, baseBit);

		unsigned int sum = exclusive_scan<add2, 0>(d_selectors, (selectorMask+1) * numElems);

		radix_sort_scatter<<<gridSizeAssign,blockSizeAssign>>>(values[1-target], values[target], numElems, d_selectors, selectorMask, baseBit);

		target = 1-target;
	}

	if(target != 0) {
		// need to copy data
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	}

	checkCudaErrors(cudaFree(d_selectors));

}

template <unsigned int(*func)(unsigned int, unsigned int), unsigned int zero>
__global__
void inclusive_scan_block(unsigned int * const d_data, const size_t numElems, unsigned int * const d_carry)
{
	extern __shared__ unsigned int s_block[];

	const size_t fullIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int * const blockStart = d_data + blockIdx.x * blockDim.x;

	if(fullIdx < numElems)
		s_block[threadIdx.x] = blockStart[threadIdx.x];
	else
		s_block[threadIdx.x] = zero;

	__syncthreads();

	int step = 1;

	while(step < blockDim.x) {
		if((threadIdx.x + step) < blockDim.x) {
			s_block[threadIdx.x + step] = func(s_block[threadIdx.x], s_block[threadIdx.x + step]);
		}
		step *= 2;

		__syncthreads();
	}

	if(fullIdx < numElems)
		blockStart[threadIdx.x] = s_block[threadIdx.x];

	if(threadIdx.x == blockDim.x-1) {
		d_carry[blockIdx.x] = s_block[threadIdx.x];
	}

}

template <unsigned int(*func)(unsigned int, unsigned int)>
__global__
void inclusive_scan_add_carry(unsigned int * const d_data, const size_t numElems, const unsigned int * const d_carry)
{
	const size_t fullIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int * const blockStart = d_data + blockIdx.x * blockDim.x;

	if(blockIdx.x > 0 && fullIdx < numElems)
		blockStart[threadIdx.x] = func(blockStart[threadIdx.x], d_carry[blockIdx.x-1]);
}



template <unsigned int(*func)(unsigned int, unsigned int), unsigned int zero>
unsigned int inclusive_scan(unsigned int * d_data, const size_t numElems)
{
	const dim3 blockSize (1024, 1, 1);
	const dim3 gridSize (numElems/blockSize.x + ((numElems%blockSize.x) == 0 ? 0 : 1), 1, 1);

	// allocate carry
	unsigned int * d_carry;
	checkCudaErrors(cudaMalloc(&d_carry, gridSize.x * sizeof(unsigned int)));

	// scan blocks
	inclusive_scan_block<func, zero> <<<gridSize, blockSize, blockSize.x*sizeof(unsigned int)>>>(d_data, numElems, d_carry);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );

	unsigned int ret;

	// recursive scan carry
	if(gridSize.x > 1) {
		ret = inclusive_scan<func, zero> (d_carry, gridSize.x);

		// add carry
		inclusive_scan_add_carry<func> <<<gridSize, blockSize>>>(d_data, numElems, d_carry);
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
	} else {
		checkCudaErrors(cudaMemcpy(&ret, d_carry, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	}


	// deallocate carry
	checkCudaErrors(cudaFree(d_carry));

	return ret;
}



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


__global__
void initHistoAccu(const unsigned int * const d_sorted_vals,
				const unsigned int numElems,
				unsigned int * const d_histo_accu)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < numElems) {
		if(idx == numElems-1 || d_sorted_vals[idx] != d_sorted_vals[idx+1]) {
			d_histo_accu[d_sorted_vals[idx]] = idx+1;
		}
	}
}

__global__
void reduceHisto(const unsigned int * const d_histo_accu,
				unsigned int * const d_histo,
				const unsigned int numBins)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < numBins) {
	  if(idx == 0) {
		  d_histo[idx] = d_histo_accu[idx];
	  } else {
		  d_histo[idx] = d_histo_accu[idx] - d_histo_accu[idx-1];
	  }
  }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
					  unsigned int* const d_histo,      //OUTPUT
					  const unsigned int numBins,
					  const unsigned int numElems)
{

	// trivial solution with atomics: 96.223869 ms
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

/*
	unsigned int * d_sorted_in;
	unsigned int * d_sorted_out;
	checkCudaErrors(cudaMalloc(&d_sorted_in, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_sorted_out, numElems * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(d_sorted_in, d_vals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	your_sort(d_sorted_in, d_sorted_out, numElems);

	unsigned int * d_histo_accu;
	checkCudaErrors(cudaMalloc(&d_histo_accu, numBins * sizeof(unsigned int)));

	{
		dim3 blockDim(1024);
		dim3 gridDim(numBins/blockDim.x + (numBins % blockDim.x == 0 ? 0 : 1) );

		histoClear<<<gridDim, blockDim>>>(d_histo_accu, numBins);
	}

	{
		dim3 blockDim(1024);
		dim3 gridDim(numElems/blockDim.x + (numElems % blockDim.x == 0 ? 0 : 1) );

		initHistoAccu<<<gridDim, blockDim>>>(d_sorted_out, numElems, d_histo_accu);
	}

	inclusive_scan<max2, 0>(d_histo_accu, numBins);

	{
		dim3 blockDim(1024);
		dim3 gridDim(numBins/blockDim.x + (numBins % blockDim.x == 0 ? 0 : 1) );

		reduceHisto<<<gridDim, blockDim>>>(d_histo_accu, d_histo, numBins);
	}

	checkCudaErrors(cudaFree(d_sorted_in));
	checkCudaErrors(cudaFree(d_sorted_out));
	checkCudaErrors(cudaFree(d_histo_accu));
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
*/
}
