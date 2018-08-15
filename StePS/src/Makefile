
#---------------------------------------------------------------------------------------------
# StePS Makefile
# Gabor Racz, 2017-2018
# 	Department of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary
# 	Department of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA
VERSION='"v0.3.2.0"'
#---------------------------------------------------------------------------------------------


#------------------------------- GPU and precision options
USING_CUDA = YES
SINGLE_PRECISION = 0

#------------------------------- Glass making options
#OPT += -DGLASS_MAKING=2048

#------------------------------- Compiler and flags
CXX = g++
BUILD_NUMBER_LDFLAGS  = -DBUILD_DATE=$$(date +'%Y%m%d') -DPROGRAM_VERSION=$(VERSION)
CFLAGS = -Wno-long-long -Wno-unused-result -lm -Wall -ansi -pedantic -O3 -ffast-math -std=gnu++98 -fopenmp -lgomp

MPI_COMPILE_FLAGS = $(shell mpic++ --showme:compile)
MPI_LINK_FLAGS = $(shell mpic++ --showme:link)

#------------------------------- Location of the MPI libraries
MPI_LIBS = /usr/lib
MPI_INC = /usr/include/mpi

ifeq ($(USING_CUDA), YES)
#------------------------------- Location of the CUDA Toolkit
CUDA_PATH       ?= /usr

NVCC = $(CUDA_PATH)/bin/nvcc -ccbin
CUDAFLAGS = -Xcompiler -fopenmp -lineinfo --compiler-options --std=gnu++98 --compiler-options -Wall --compiler-options -ansi -O3 -lm -Xcompiler -pthread -Xcompiler \'-Wl\\,-rpath\' -Xcompiler \'-Wl\\,$(MPI_LIBS)\' -Xcompiler \'-Wl\\,--enable-new-dtags\' -lmpi_cxx -lmpi
CUDA_INC = -I$(CUDA_PATH)/include

SRC = main.cc forces_cuda.cu ewald_space.cc step.cc read_paramfile.cc friedmann_solver.cc read_gadget_ic.cc
OBJ = main.o forces_cuda.o ewald_space.o step.o read_paramfile.o friedmann_solver.o read_gadget_ic.o
DEPS = global_variables.h
PROG = StePS_CUDA

ifeq ($(SINGLE_PRECISION), 1)
OPT += -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION)
$(PROG): $(OBJ)
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ)
%.o: %.cc
	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) $(OPT) -o $@ -c $<
else
$(PROG): $(OBJ)
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ)
%.o: %.cc
	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) -L$(MPI_LIBS) -I$(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) $(OPT) -o $@ -c $<
endif
clean:
	rm -fv $(OBJ) $(PROG)

else

SRC = main.cc forces.cc ewald_space.cc step.cc read_paramfile.cc friedmann_solver.cc read_gadget_ic.cc
OBJ = main.o forces.o ewald_space.o step.o read_paramfile.o friedmann_solver.o read_gadget_ic.o
DEPS = global_variables.h
PROG = StePS
ifeq ($(SINGLE_PRECISION), 1)
OPT += -DUSE_SINGLE_PRECISION=$(SINGLE_PRECISION)
$(PROG): $(OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ)

%.o: %.cc
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<
else
$(PROG): $(OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ)

%.o: %.cc
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(E_INC) $(D_INC) $(E_LIB) $(D_LIB) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $< 

endif

clean:
	rm -fv $(OBJ) $(PROG)

endif
