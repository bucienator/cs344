//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
		For example [0 0 1 1 0 0 1]
				->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
	  output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__
void exclusive_scan_add_carry(unsigned int * const d_data, const size_t numElems, const unsigned int * const d_carry)
{
	const size_t fullIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int * const blockStart = d_data + blockIdx.x * blockDim.x;

	if(fullIdx < numElems)
		blockStart[threadIdx.x] += d_carry[blockIdx.x];
}


__global__
void exclusive_scan_block(unsigned int * const d_data, const size_t numElems, unsigned int * const d_carry)
{
	extern __shared__ unsigned int s_block[];

	const size_t fullIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int * const blockStart = d_data + blockIdx.x * blockDim.x;

	if(fullIdx < numElems)
		s_block[threadIdx.x] = blockStart[threadIdx.x];
	else
		s_block[threadIdx.x] = 0;

	if(threadIdx.x == blockDim.x-1) {
		d_carry[blockIdx.x] = s_block[threadIdx.x];
	}

	__syncthreads();

	int step = 1;

	while(step < blockDim.x) {
		if(threadIdx.x % (step * 2) == 0 && (threadIdx.x + step) < blockDim.x) {
			s_block[blockDim.x - 1 - threadIdx.x] += s_block[blockDim.x - 1 - threadIdx.x - step];
		}
		step *= 2;

		__syncthreads();
	}

	s_block[blockDim.x - 1] = 0;
	__syncthreads();

	do {
		step /= 2;
		if(threadIdx.x % (step * 2) == 0 && (threadIdx.x + step) < blockDim.x) {
			unsigned int temp = s_block[blockDim.x - 1 - threadIdx.x];
			s_block[blockDim.x - 1 - threadIdx.x] += s_block[blockDim.x - 1 - threadIdx.x - step];
			s_block[blockDim.x - 1 - threadIdx.x - step] = temp;
		}
		__syncthreads();
	} while(step > 1);

	if(fullIdx < numElems)
		blockStart[threadIdx.x] = s_block[threadIdx.x];

	if(threadIdx.x == blockDim.x-1) {
		d_carry[blockIdx.x] += s_block[threadIdx.x];
	}

}


void exclusive_scan(unsigned int * d_data, const size_t numElems)
{
	const dim3 blockSize (1024, 1, 1);
	const dim3 gridSize (numElems/blockSize.x + ((numElems%blockSize.x) == 0 ? 0 : 1), 1, 1);

	// allocate carry
	unsigned int * d_carry;
	checkCudaErrors(cudaMalloc(&d_carry, gridSize.x * sizeof(unsigned int)));

	// scan blocks
	exclusive_scan_block<<<gridSize, blockSize, blockSize.x*sizeof(unsigned int)>>>(d_data, numElems, d_carry);
	//gpuErrchk( cudaPeekAtLastError() );
	//gpuErrchk( cudaDeviceSynchronize() );

	// recursive scan carry
	if(gridSize.x > 1) {
		exclusive_scan(d_carry, gridSize.x);

		// add carry
		exclusive_scan_add_carry<<<gridSize, blockSize>>>(d_data, numElems, d_carry);
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
	}


	// deallocate carry
	checkCudaErrors(cudaFree(d_carry));

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
					   const unsigned int* const d_inputPos,
					   unsigned int* const d_outputVals,
					   unsigned int* const d_outputPos,
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
		d_outputPos[pos] = d_inputPos[idx];
	}

}


void your_sort(unsigned int* const d_inputVals,
			   unsigned int* const d_inputPos,
			   unsigned int* const d_outputVals,
			   unsigned int* const d_outputPos,
			   const size_t numElems)
{
  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code MUST RUN BEFORE YOUR CODE in case you accidentally change       *
  * the input values when implementing your radix sort.                       *
  *                                                                           *
  * This code performs the reference radix sort on the host and compares your *
  * sorted values to the reference.                                           *
  *                                                                           *
  * Thrust containers are used for copying memory from the GPU                *
  * ************************************************************************* */
  /*
  thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(d_inputVals),
												thrust::device_ptr<unsigned int>(d_inputVals) + numElems);
  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(d_inputPos),
											   thrust::device_ptr<unsigned int>(d_inputPos) + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
						&h_outputVals[0], &h_outputPos[0],
						numElems);
  */
	const unsigned int baseBitIncrement = 4;
	const unsigned int selectorMask = (1 << baseBitIncrement) - 1;

	unsigned int * d_selectors;
	checkCudaErrors(cudaMalloc(&d_selectors, (selectorMask+1) * numElems * sizeof(unsigned int)));
  
	const dim3 blockSizeAssign (1024, 1, 1);
	const dim3 gridSizeAssign (numElems/blockSizeAssign.x + ((numElems%blockSizeAssign.x) == 0 ? 0 : 1), 1, 1);

	unsigned int * values[] = {d_inputVals, d_outputVals};
	unsigned int * pos[] = {d_inputPos, d_outputPos};
	int target = 1;
	for(unsigned int baseBit = 0; baseBit < sizeof(unsigned int) * 8; baseBit += baseBitIncrement) {
		radix_sort_assign<<<gridSizeAssign,blockSizeAssign>>>(values[1-target], d_selectors, numElems, selectorMask, baseBit);
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
		/*{
			thrust::host_vector<unsigned int> h_test(thrust::device_ptr<unsigned int>(d_selectors), thrust::device_ptr<unsigned int>(d_selectors + (selectorMask+1) * numElems));
			unsigned int accu = 0;
			size_t elemCnt = h_test.size();
			for(size_t i = 0; i < h_test.size(); i++) {
				unsigned int h = h_test[i];
				if(h_test[i]!= 0 && h_test[i] != 1) {
					printf("test error 3");
					exit(1);
				}
				accu += h_test[i];
			}
			if(accu != numElems) {
					printf("test error 4");
					exit(1);
			}
		}*/

		exclusive_scan(d_selectors, (selectorMask+1) * numElems);
		/*{
			thrust::host_vector<unsigned int> h_test(thrust::device_ptr<unsigned int>(d_selectors), thrust::device_ptr<unsigned int>(d_selectors + (selectorMask+1) * numElems));
			unsigned int lastValue = 0;
			for(size_t i = 0; i < h_test.size(); i++) {
				if(lastValue > h_test[i]) {
					printf("test error");
					exit(1);
				}
				lastValue = h_test[i];
			}
			if(lastValue != numElems-1 && lastValue != numElems) {
					printf("test error 2");
					exit(1);
			}
		}*/

		radix_sort_scatter<<<gridSizeAssign,blockSizeAssign>>>(values[1-target], pos[1-target], values[target], pos[target], numElems, d_selectors, selectorMask, baseBit);
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );

		target = 1-target;
	}

	if(target != 0) {
		// need to copy data
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	}

	checkCudaErrors(cudaFree(d_selectors));

  /* *********************************************************************** *
   * Uncomment the code below to do the correctness checking between your    *
   * result and the reference.                                               *
   **************************************************************************/

  /*
  thrust::host_vector<unsigned int> h_yourOutputVals(thrust::device_ptr<unsigned int>(d_outputVals),
													 thrust::device_ptr<unsigned int>(d_outputVals) + numElems);
  thrust::host_vector<unsigned int> h_yourOutputPos(thrust::device_ptr<unsigned int>(d_outputPos),
													thrust::device_ptr<unsigned int>(d_outputPos) + numElems);

  checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
  checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);
  */
}
