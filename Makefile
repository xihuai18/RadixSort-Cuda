CU_APPS=radixSort    # specify the name of CUDA file (exclude .cu)

GCC_HOME=/home/jovyan/gcc6/bin/g++    # specify your own gcc install path

NVCC_FLAGS  = -O3

.PHONY : clean

all: ${CU_APPS}

${CU_APPS}: kernel_radix.cu kernel_exclusiveScan.cu main.cu
	nvcc $(NVCC_FLAGS) kernel_radix.cu kernel_exclusiveScan.cu main.cu -o $@ 

clean:
	rm -f ${CU_APPS}