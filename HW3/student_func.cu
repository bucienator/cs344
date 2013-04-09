/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>

#include "reference_calc.cpp"
#include "utils.h"

__global__
void cdf(unsigned int * const d_cdf,
         const int numBins)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int step = 1;
    while(step < numBins) {
        if(idx % (step*2) == 0 && (numBins-1-idx - step) < numBins) {
            d_cdf[numBins-1-idx] += d_cdf[numBins-1-idx - step];
        }
        step *= 2;
        __syncthreads();
    }

    step /= 2;
    if(idx == numBins -1) {
        d_cdf[idx] = 0;
    }
    __syncthreads();

    while(step) {
        if(idx % (step*2) == 0 && numBins-1-idx - step < numBins) {
            int tmp = d_cdf[numBins-1-idx];
            d_cdf[numBins-1-idx] += d_cdf[numBins-1-idx - step];
            d_cdf[numBins-1-idx - step] = tmp;
        }
        step /= 2;
        __syncthreads();
    }

}

__global__
void histogram(const float * const d_logLuminance,
               unsigned int * const d_hist,
               const float lumMin,
               const float lumRange,
               const int numBins,
               const size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numBins) {
        d_hist[idx] = 0;
    }
    
    __syncthreads();
    
    if(idx < N) {
        size_t bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_logLuminance[idx] - lumMin) / lumRange * numBins));
        atomicAdd(&d_hist[bin], 1);
    }
}

__global__
void findMinAndMax(const float * const d_logLuminance,
                   float * const d_work,
                   const size_t N)
{
    // calculate min/max in block
    size_t idx = threadIdx.x;
    const float * const d_base = &d_logLuminance[blockIdx.x * blockDim.x];
    float * const d_workbase = &d_work[blockIdx.x * blockDim.x];
    size_t BN = blockIdx.x == gridDim.x-1 ? N % blockDim.x : blockDim.x;

    int step = 1;
    
    // unrolling for step == 1, combined min-max
    if(idx % 2 == 0) {
        if(idx + step < BN && d_base[idx] > d_base[idx+step]) {
            d_workbase[idx] = d_base[idx+step];
            d_workbase[idx+step] = d_base[idx];
        } else {
            d_workbase[idx] = d_base[idx];
            if(idx + step < BN) {
                d_workbase[idx+step] = d_base[idx+step];
            }
        }
    }
    step *= 2;
    __syncthreads();

    while(step < BN) {
        if(idx % (step*2) == 0 && idx < BN) {
            // calculate minimum
            if(idx + step < BN && d_workbase[idx] > d_workbase[idx+step]) {
                d_workbase[idx] = d_workbase[idx+step];
            }
        }
        
        if((idx-1) % (step*2) == 0 && idx < BN) {
            // calculate maximum
            size_t cmpIdx = idx + step == BN ? idx + step - 1 : idx + step;
            if(cmpIdx < BN && d_workbase[idx] < d_workbase[cmpIdx]) {
                d_workbase[idx] = d_workbase[cmpIdx];
            }
        }
        step *= 2;
        __syncthreads();
    }
    
 }

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    const dim3 blockSize (1024, 1, 1);
    const size_t elemCount = numRows * numCols;
    const dim3 gridSize (elemCount/blockSize.x + ((elemCount%blockSize.x) == 0 ? 0 : 1), 1, 1);
    
    float * d_work;
    checkCudaErrors(cudaMalloc(&d_work, sizeof(float) * elemCount));
    
    findMinAndMax<<<gridSize, blockSize>>>(d_logLuminance, d_work, elemCount);
    
    // min/max are in start of blocks
    float * h_work = new float[elemCount];
    cudaMemcpy(h_work, d_work, sizeof(float)*elemCount, cudaMemcpyDeviceToHost);
    cudaFree(d_work);
    
    // serial look for block start
    min_logLum = h_work[0];
    max_logLum = h_work[1];
    for(int i = 1024; i < elemCount; i++) {
        min_logLum = min(min_logLum, h_work[i]);
        max_logLum = max(max_logLum, h_work[i+1]);
    }
    
    //printf("min %f  max %f\n", min_logLum, max_logLum);
    float lumRange = max_logLum - min_logLum;
    
    delete [] h_work;
    
    histogram<<<gridSize, blockSize>>>(d_logLuminance, d_cdf, min_logLum, lumRange, numBins, elemCount);
    cdf<<<1,numBins>>>(d_cdf, numBins);

    /*
    unsigned int * h_hist = new unsigned int[numBins];
    cudaMemcpy(h_hist, d_cdf, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToHost);
    for(int i = 0; i < numBins; i++) {
        printf("%u\n", h_hist[i]);
    }
    delete [] h_hist;
    */
    
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */


}
