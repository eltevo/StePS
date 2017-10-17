USING_CUDA = YES
SINGLE_PRECISION = 1

CXX = g++
BUILD_NUMBER_LDFLAGS  = -Xlinker --defsym -Xlinker __BUILD_DATE=$$(date +'%Y%m%d')
CFLAGS = -lm -Wall -ansi -pedantic -O3 -ffast-math -std=c++11 -fopenmp

ifeq ($(USING_CUDA), YES)
# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda-7.5

NVCC = $(CUDA_PATH)/bin/nvcc -ccbin
CUDAFLAGS = -lineinfo --compiler-options --std=c++11 --compiler-options -Wall --compiler-options -ansi -O3 -Xcompiler -fopenmp -lm
CUDA_INC = -I$(CUDA_PATH)/include

SRC = main.cc forces_cuda.cu ewald_space.cc step.cc read_paramfile.cc friedmann_solver.cc read_gadget_ic.cc
OBJ = main.o forces_cuda.o ewald_space.o step.o read_paramfile.o friedmann_solver.o read_gadget_ic.o
DEPS = global_variables.h
PROG = StePS_CUDA

ifeq ($(SINGLE_PRECISION), 1)
$(PROG): $(OBJ)
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(BUILD_NUMBER_LDFLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $(PROG) $(OBJ)
%.o: %.cc
	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CXX) $(CUDA_INC) $(BUILD_NUMBER_LDFLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $@ -c $<
else
$(PROG): $(OBJ)
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(BUILD_NUMBER_LDFLAGS) -o $(PROG) $(OBJ)
%.o: %.cc
	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CXX) $(CUDA_INC) $(BUILD_NUMBER_LDFLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) -o $@ -c $<
endif
clean:
	rm -fv $(OBJ) $(PROG)

else

SRC = main.cc forces.cc ewald_space.cc step.cc read_paramfile.cc friedmann_solver.cc read_gadget_ic.cc
OBJ = main.o forces.o ewald_space.o step.o read_paramfile.o friedmann_solver.o read_gadget_ic.o
DEPS = global_variables.h
PROG = StePS
ifeq ($(SINGLE_PRECISION), 1)
$(PROG): $(OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) $(BUILD_NUMBER_LDFLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $(PROG) $(OBJ)

%.o: %.cc
	$(CXX) $(CFLAGS) $(LDFLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $@ -c $<
else
$(PROG): $(OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) $(BUILD_NUMBER_LDFLAGS) -o $(PROG) $(OBJ)

%.o: %.cc
	$(CXX) $(CFLAGS) $(LDFLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) -o $@ -c $< 

endif

clean:
	rm -fv $(OBJ) $(PROG)

endif
