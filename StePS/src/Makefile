USING_CUDA = NO
SINGLE_PRECISION = 0

CXX = g++
BUILD_NUMBER_LDFLAGS  = -Xlinker --defsym -Xlinker __BUILD_DATE=$$(date +'%Y%m%d')
CFLAGS = -Wno-long-long -lm -Wall -ansi -pedantic -O3 -ffast-math -std=gnu++98 -fopenmp -lgomp

MPI_COMPILE_FLAGS = $(shell mpic++ --showme:compile)
MPI_LINK_FLAGS = $(shell mpic++ --showme:link)

MPI_LIBS = /usr/lib64/openmpi/lib
MPI_INC = /usr/include/openmpi-x86_64

ifeq ($(USING_CUDA), YES)
# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda-9.1

NVCC = $(CUDA_PATH)/bin/nvcc -ccbin
CUDAFLAGS = -Xcompiler -fopenmp -lineinfo --compiler-options --std=gnu++98 --compiler-options -Wall --compiler-options -ansi -O3 -lm -Xcompiler -pthread -Xcompiler \'-Wl\\,-rpath\' -Xcompiler \'-Wl\\,$(MPI_LIBS)\' -Xcompiler \'-Wl\\,--enable-new-dtags\' -lmpi_cxx -lmpi
CUDA_INC = -I$(CUDA_PATH)/include

SRC = main.cc forces_cuda.cu ewald_space.cc step.cc read_paramfile.cc friedmann_solver.cc read_gadget_ic.cc
OBJ = main.o forces_cuda.o ewald_space.o step.o read_paramfile.o friedmann_solver.o read_gadget_ic.o
DEPS = global_variables.h
PROG = StePS_CUDA

ifeq ($(SINGLE_PRECISION), 1)
$(PROG): $(OBJ)
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $(PROG) $(OBJ)
%.o: %.cc
	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $@ -c $<
else
$(PROG): $(OBJ)
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) -o $(PROG) $(OBJ)
%.o: %.cc
	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) -o $@ -c $<
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
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $(PROG) $(OBJ)

%.o: %.cc
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION) -o $@ -c $<
else
$(PROG): $(OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) -o $(PROG) $(OBJ)

%.o: %.cc
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) -o $@ -c $< 

endif

clean:
	rm -fv $(OBJ) $(PROG)

endif
