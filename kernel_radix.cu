/*
name: Xihuai Wang
student_id: 16337236
file description:
  kernels and api for radix sort
*/

#include "common.h"
#include "hw3_cuda.h"

/*
generate mask for elements
  @One:
    if true: mask has 1 for elements that have 1 in the @bit_shift th bit
    else: ....
*/
__global__ void getMask(unsigned int *d_in, unsigned int *d_out, unsigned int in_size, unsigned int bit_shift,
                         unsigned int One) {
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int bit = 0;
  if (index < in_size) {
    bit = d_in[index] & (1 << bit_shift);
    bit = (bit > 0) ? 1 : 0;
    d_out[index] = (One ? bit : 1 - bit);
  }
}

/*
calculate indices for elements
  @total_pre: the num of elements in the front buckets 
*/
__global__ void getIndex(unsigned int *d_index, unsigned int *d_scan, unsigned int *d_mask, unsigned int in_size,
                            unsigned int total_pre) {
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index < in_size) {
    if (d_mask[index] == 1) {
      d_index[index] = total_pre + d_scan[index];
    }
  }
}

/*
scatter the elements to specific buckets
*/
__global__ void scatter(unsigned int *d_in, unsigned int *d_index, unsigned int *d_out,
                                unsigned int in_size) {
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < in_size) {
    d_out[d_index[index]] = d_in[index];
  }
}

/*
api for radix sort
*/
void radix_sort(unsigned int **d_in, unsigned int **d_out, unsigned int *d_out_scan, unsigned int *h_in,
                unsigned int *h_out_scan, unsigned int num_elements) {
  unsigned int *d_temp;
  unsigned int *d_index;

  unsigned int scan_last, mask_last;
  unsigned int total_zeros;

  dim3 Block(BLOCK_SIZE, 1, 1);

  dim3 Grid((unsigned int)(ceil(num_elements / (1.0 * Block.x))), 1, 1);

  CHECK(cudaMalloc((void **)&d_index, num_elements * sizeof(unsigned int)));

  for (unsigned int i = 0; i < 32; i++) {
    getMask<<<Grid, Block>>>(*d_in, *d_out, num_elements, i, false);
    // exclusive scan
    exclusiveScan(d_out_scan, *d_out, num_elements);
    CHECK(cudaDeviceSynchronize());
    
    // get the number of the elements that have 0 at the @i th bit
    CHECK(cudaMemcpy(&scan_last, d_out_scan + num_elements - 1, sizeof(unsigned int),
    cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&mask_last, *d_out + num_elements - 1, sizeof(unsigned int),
    cudaMemcpyDeviceToHost));
    
    total_zeros = scan_last + (mask_last == 1 ? 1 : 0);
    
    getIndex<<<Grid, Block>>>(d_index, d_out_scan, *d_out,
      num_elements, 0);
      
    getMask<<<Grid, Block>>>(*d_in, *d_out, num_elements, i, true);
      
    // exclusive scan
    exclusiveScan(d_out_scan, *d_out, num_elements);

    getIndex<<<Grid, Block>>>(d_index, d_out_scan, *d_out,
                                             num_elements, total_zeros);
    scatter<<<Grid, Block>>>(*d_in, d_index, *d_out,
                                                 num_elements);

    // swap pointers
    d_temp = *d_in;
    *d_in = *d_out;
    *d_out = d_temp;
  }
  *d_out = *d_in;
  cudaFree(d_index);
}
