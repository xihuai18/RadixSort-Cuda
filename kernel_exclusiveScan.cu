/*
exclusive scan using Blelloch Scan
@sum: 
sum of the elements in a block
*/
#include "hw3_cuda.h"
#include "common.h"

// texture<unsigned int, 1, cudaReadModeElementType> tex_sum;
const int MAX_CONSTANT = 16*1024;
__constant__ unsigned int const_sum[MAX_CONSTANT];

// pay attention that blockDim.x must be power of 2
__global__ void blellochScan(unsigned int *out, unsigned int *in,
                              unsigned int *sum, unsigned int inputSize) {
  __shared__ unsigned int temp[2 * BLOCK_SIZE];
  unsigned int start = blockIdx.x * blockDim.x << 1;
  unsigned int tx = threadIdx.x;
  unsigned int index = 0;
  temp[tx] = (start + tx < inputSize)? in[start+tx]:0;

  temp[tx+blockDim.x] = (start + tx + blockDim.x < inputSize)? in[start + tx + blockDim.x] : 0;

  // Blelloch Scan
  __syncthreads();
  // reduction step
  unsigned int stride = 1;
  while (stride <= blockDim.x) {
    index = (tx + 1) * (stride << 1) - 1;
    if (index < (blockDim.x << 1)) {
      temp[index] += temp[index - stride];
    }
    stride <<= 1;
    __syncthreads();
  }
  // first store the reduction sum in sum array
  // make it zero since it is exclusive scan
  if (tx == 0) {
    // sum array contains the prefix sum of each
    // 2*blockDim blocks of element.
    if (sum != NULL) {
      sum[blockIdx.x] = temp[(blockDim.x << 1) - 1];
    }
    temp[(blockDim.x << 1) - 1] = 0;
  }
  // wait for thread zero to write
  __syncthreads();
  // post scan step
  stride = blockDim.x;
  index = 0;
  unsigned int var = 0;
  while (stride > 0) {
    index = ((stride << 1) * (tx + 1)) - 1;
    if (index < (blockDim.x << 1)) {
      var = temp[index];
      temp[index] += temp[index - stride];
      temp[index - stride] = var;
    }
    stride >>= 1;
    __syncthreads();
  }

  // now write the temp array to output
  if (start + tx < inputSize) {
    out[start + tx] = temp[tx];
  }
  if (start + tx + blockDim.x < inputSize) {
    out[start + tx + blockDim.x] = temp[tx + blockDim.x];
  }
}

/* 
sum out the blocks' accumulated sums to each element
*/
__global__ void mergeScanBlocks(unsigned int *sum, unsigned int *output,
                                unsigned int opSize) {
  unsigned int index = (blockDim.x * blockIdx.x << 1) + threadIdx.x;
  if (index < opSize) {
    // output[index] += sum[blockIdx.x];
    output[index] += (opSize > MAX_CONSTANT)? sum[blockIdx.x]:const_sum[blockIdx.x];
    // output[index] += tex1Dfetch(tex_sum, blockIdx.x);
  }
  if (index + blockDim.x < opSize) {
    // output[index + blockDim.x] += sum[blockIdx.x];
    output[index + blockDim.x] += (opSize > MAX_CONSTANT)? sum[blockIdx.x]:const_sum[blockIdx.x];
    // output[index + blockDim.x] += tex1Dfetch(tex_sum, blockIdx.x);
  }
}

/* 
api for exclusiveScan
*/
void exclusiveScan(unsigned int *out, unsigned int *in, unsigned int in_size) {
  unsigned int numBlocks1 = in_size / BLOCK_SIZE;
  if (in_size % BLOCK_SIZE) numBlocks1++;
  unsigned int numBlocks2 = numBlocks1 / 2;
  if (numBlocks1 % 2) numBlocks2++;
  dim3 dimThreadBlock;
  dimThreadBlock.x = BLOCK_SIZE;
  dimThreadBlock.y = 1;
  dimThreadBlock.z = 1;
  dim3 dimGrid;
  dimGrid.x = numBlocks2;
  dimGrid.y = 1;
  dimGrid.z = 1;

  unsigned int *d_sumArr = NULL;
  if (in_size > (2 * BLOCK_SIZE)) {
    // we need the sum auxilarry  array only if nuFmblocks2 > 1
    CHECK(cudaMalloc((void **)&d_sumArr, numBlocks2 * sizeof(unsigned int)));
  }
  blellochScan<<<dimGrid, dimThreadBlock>>>(out, in, d_sumArr, in_size);

  if (in_size <= (2 * BLOCK_SIZE)) {
    // out has proper exclusive scan. just return
    CHECK(cudaDeviceSynchronize());
    return;
  } else {
    // now we need to perform exclusive scan on the auxilliary sum array
    unsigned int *d_sumArr_scan;
    CHECK(cudaMalloc((void **)&d_sumArr_scan, numBlocks2 * sizeof(unsigned int)));
    exclusiveScan(d_sumArr_scan, d_sumArr, numBlocks2);
    // d_sumArr_scan now contains the exclusive scan op of individual blocks
    // now just do a one-one addition of blocks
    // cudaBindTexture(0, tex_sum, d_sumArr_scan, numBlocks2 * sizeof(unsigned int));
    if(numBlocks2 <= MAX_CONSTANT) {
      CHECK(cudaMemcpyToSymbol(const_sum, d_sumArr_scan, numBlocks2 * sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice));
    }
    mergeScanBlocks<<<dimGrid, dimThreadBlock>>>(d_sumArr_scan, out, in_size);
    // cudaUnbindTexture(tex_sum);
    
    cudaFree(d_sumArr);
    cudaFree(d_sumArr_scan);
  }
}
