//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
	  as boundary conditions for solving a Poisson equation that tells
	  us how to blend the images.
   
	  No pixels from the destination except pixels on the border
	  are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
	  Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
			 else if the neighbor in on the border then += DestinationImg[neighbor]

	  Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
	  float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
	  ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


	In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

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
void classify_pixels(const uchar4 * const d_sourceImg,
					const size_t numRowsSource, const size_t numColsSource,
					unsigned int * const d_borderMask,
					unsigned int * const d_interiorMask)
{
	const size_t pixelCount = numRowsSource * numColsSource;
	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < pixelCount; i += gridDim.x * blockDim.x) {

		d_interiorMask[i] = 0;
		d_borderMask[i] = 0;

		uchar4 me = d_sourceImg[i];

		if(me.x != 255 || me.y != 255 || me.z != 255) {
			bool isTopInterior = true;
			if(i - numColsSource < pixelCount) {
				uchar4 pix = d_sourceImg[i-numColsSource];
				if(pix.x == 255 && pix.y == 255 && pix.z == 255) {
					isTopInterior = false;
				}
			}
			bool isLeftInterior = true;
			if(i - 1 < pixelCount) {
				uchar4 pix = d_sourceImg[i-1];
				if(pix.x == 255 && pix.y == 255 && pix.z == 255) {
					isLeftInterior = false;
				}
			}
			bool isRightInterior = true;
			if(i + 1 < pixelCount) {
				uchar4 pix = d_sourceImg[i+1];
				if(pix.x == 255 && pix.y == 255 && pix.z == 255) {
					isRightInterior = false;
				}
			}
			bool isBottomInterior = true;
			if(i + numColsSource < pixelCount) {
				uchar4 pix = d_sourceImg[i+numColsSource];
				if(pix.x == 255 && pix.y == 255 && pix.z == 255) {
					isBottomInterior = false;
				}
			}
	
			if(!isTopInterior || !isLeftInterior || !isRightInterior || !isBottomInterior) {
				d_borderMask[i] = 1;
			} else {
				d_interiorMask[i] = 1;
			}
		}
	}
}

__global__
void compact(const unsigned int * const d_mask,
			const unsigned int * const d_positions,
			const size_t size,
			unsigned int * const d_target)
{
	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
		if(d_mask[i] == 1) {
			d_target[d_positions[i]] = i;
		}
	}
}

__global__
void init_blended(const uchar4 * const d_sourceImg,
				const uchar4 * const d_destImg,
				uchar4 * const d_blendedImg,
				const unsigned int * const d_borderMask,
				const unsigned int * const d_interiorMask,
				const size_t size)
{
	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
		if(d_interiorMask[i] == 1) {
			d_blendedImg[i] = d_sourceImg[i];
		} else {
			d_blendedImg[i] = d_destImg[i];
		}
	}
}

__global__
void split_channels(const uchar4 * const d_in,
					float * const ch1,
					float * const ch2,
					float * const ch3,
					const size_t size)
{
	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
		ch1[i] = d_in[i].x;
		ch2[i] = d_in[i].y;
		ch3[i] = d_in[i].z;
	}
}


/*
   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
	  Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
			 else if the neighbor in on the border then += DestinationImg[neighbor]

	  Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
	  float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
	  ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
*/
__global__
void iterate(const float * const src,
			const float * const in,
			float * const out,
			const size_t numRowsSource, const size_t numColsSource,
			const unsigned int * const d_borderMask,
			const unsigned int * const d_interiorIdxs,
			const size_t count)
{
	const size_t pixelCount = numRowsSource * numColsSource;

	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
		unsigned int idx = d_interiorIdxs[i];
		float sum1 = .0f;
		float sum2 = .0f;

		if(idx - numColsSource < pixelCount) {
			sum1 += in[idx - numColsSource];
			sum2 += src[idx] - src[idx - numColsSource];
		}
		if(idx - 1 < pixelCount) {
			sum1 += in[idx - 1];
			sum2 += src[idx] - src[idx - 1];
		}
		if(idx + 1 < pixelCount) {
			sum1 += in[idx + 1];
			sum2 += src[idx] - src[idx + 1];
		}
		if(idx + numColsSource < pixelCount) {
			sum1 += in[idx + numColsSource];
			sum2 += src[idx] - src[idx + numColsSource];
		}
		float val = (sum1+sum2) / 4.f;
		if(val > 255) val = 255;
		if(val < 0) val = 0;
		out[idx] = val;
	}
}

__global__
void blend_channels(uchar4 * const d_blendedImg,
					const float * const ch1,
					const float * const ch2,
					const float * const ch3,
			const unsigned int * const d_interiorIdxs,
			const size_t count)
{
	for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
		unsigned int idx = d_interiorIdxs[i];

		d_blendedImg[idx].x = ch1[idx];
		d_blendedImg[idx].y = ch2[idx];
		d_blendedImg[idx].z = ch3[idx];
	}
}


void your_blend(const uchar4* const h_sourceImg,  //IN
				const size_t numRowsSource, const size_t numColsSource,
				const uchar4* const h_destImg, //IN
				uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
	 1) Compute a mask of the pixels from the source image to be copied
		The pixels that shouldn't be copied are completely white, they
		have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

	 2) Compute the interior and border regions of the mask.  An interior
		pixel has all 4 neighbors also inside the mask.  A border pixel is
		in the mask itself, but has at least one neighbor that isn't.

	 3) Separate out the incoming image into three separate channels

	 4) Create two float(!) buffers for each color channel that will
		act as our guesses.  Initialize them to the respective color
		channel of the source image since that will act as our intial guess.

	 5) For each color channel perform the Jacobi iteration described 
		above 800 times.

	 6) Create the output image by replacing all the interior pixels
		in the destination image with the result of the Jacobi iterations.
		Just cast the floating point values to unsigned chars since we have
		already made sure to clamp them to the correct range.

	  Since this is final assignment we provide little boilerplate code to
	  help you.  Notice that all the input/output pointers are HOST pointers.

	  You will have to allocate all of your own GPU memory and perform your own
	  memcopies to get data in and out of the GPU memory.

	  Remember to wrap all of your calls with checkCudaErrors() to catch any
	  thing that might go wrong.  After each kernel call do:

	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	  to catch any errors that happened while executing the kernel.
  */


	uchar4 * d_sourceImg;
	checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));
	uchar4 * d_destImg;
	checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));

	// classify pixels: border, interior
	unsigned int * d_borderMask;
	checkCudaErrors(cudaMalloc(&d_borderMask, sizeof(unsigned int) * numRowsSource * numColsSource));
	unsigned int * d_interiorMask;
	checkCudaErrors(cudaMalloc(&d_interiorMask, sizeof(unsigned int) * numRowsSource * numColsSource));

	classify_pixels<<<32,32>>>(d_sourceImg, numRowsSource, numColsSource, d_borderMask, d_interiorMask);

	uchar4 * d_blendedImg;
	checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4) * numRowsSource * numColsSource));

	init_blended<<<32,32>>>(d_sourceImg, d_destImg, d_blendedImg, d_borderMask, d_interiorMask, numRowsSource * numColsSource);

	unsigned int * d_borderMask2;
	checkCudaErrors(cudaMalloc(&d_borderMask2, sizeof(unsigned int) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemcpy(d_borderMask2, d_borderMask, sizeof(unsigned int) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
	unsigned int * d_interiorMask2;
	checkCudaErrors(cudaMalloc(&d_interiorMask2, sizeof(unsigned int) * numRowsSource * numColsSource));
	checkCudaErrors(cudaMemcpy(d_interiorMask2, d_interiorMask, sizeof(unsigned int) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));

	//unsigned int borderCnt = exclusive_scan<add2, 0>(d_borderMask2, numRowsSource * numColsSource);
	unsigned int interiorCnt = exclusive_scan<add2, 0>(d_interiorMask2, numRowsSource * numColsSource);

	//unsigned int * d_borderIdxs;
	//checkCudaErrors(cudaMalloc(&d_borderIdxs, sizeof(unsigned int) * borderCnt));
	unsigned int * d_interiorIdxs;
	checkCudaErrors(cudaMalloc(&d_interiorIdxs, sizeof(unsigned int) * interiorCnt));

	//compact<<<32,32>>>(d_borderMask, d_borderMask2, numRowsSource * numColsSource, d_borderIdxs);
	compact<<<32,32>>>(d_interiorMask, d_interiorMask2, numRowsSource * numColsSource, d_interiorIdxs);

	checkCudaErrors(cudaFree(d_interiorMask2));
	checkCudaErrors(cudaFree(d_interiorMask));
	//checkCudaErrors(cudaFree(d_borderMask2));

	float * buf[9];
	for(int i = 0; i < 9; i++) {
		checkCudaErrors(cudaMalloc(&buf[i], sizeof(float) * numRowsSource * numColsSource));
	}
	split_channels<<<32,32>>>(d_blendedImg, buf[0], buf[1], buf[2], numRowsSource * numColsSource);
	split_channels<<<32,32>>>(d_blendedImg, buf[3], buf[4], buf[5], numRowsSource * numColsSource);
	split_channels<<<32,32>>>(d_sourceImg, buf[6], buf[7], buf[8], numRowsSource * numColsSource);

	int target = 1;
	for(int i = 0; i < 800; i++, target = 1 - target) {
		iterate<<<32,32>>>(buf[6 + 0], buf[(1-target)*3 + 0], buf[target*3 + 0], numRowsSource, numColsSource, d_borderMask, d_interiorIdxs, interiorCnt);
		iterate<<<32,32>>>(buf[6 + 1], buf[(1-target)*3 + 1], buf[target*3 + 1], numRowsSource, numColsSource, d_borderMask, d_interiorIdxs, interiorCnt);
		iterate<<<32,32>>>(buf[6 + 2], buf[(1-target)*3 + 2], buf[target*3 + 2], numRowsSource, numColsSource, d_borderMask, d_interiorIdxs, interiorCnt);
	}

	blend_channels<<<32,32>>>(d_blendedImg, buf[(1-target)*3 + 0], buf[(1-target)*3 + 1], buf[(1-target)*3 + 2], d_interiorIdxs, interiorCnt);

	// copy results
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyDeviceToHost));

	// free buffers
	for(int i = 0; i < 9; i++) {
		checkCudaErrors(cudaFree(buf[i]));
	}
	checkCudaErrors(cudaFree(d_borderMask));
	checkCudaErrors(cudaFree(d_blendedImg));
	checkCudaErrors(cudaFree(d_destImg));
	checkCudaErrors(cudaFree(d_sourceImg));


}
