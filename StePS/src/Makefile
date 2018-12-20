
#-----------------------------------------------------------------------------------------------#
# StePS Makefile										#
# Copyright (C) 2018 Gabor Racz									#
# 	Department of Physics of Complex Systems, Eotvos Lorand University | Budapest, Hungary	#
# 	Department of Physics & Astronomy, Johns Hopkins University | Baltimore, MD, USA	#
VERSION='"v0.3.7.0"'#										#
#    This program is free software; you can redistribute it and/or modify			#
#    it under the terms of the GNU General Public License as published by			#
#    the Free Software Foundation; either version 2 of the License, or				#
#    (at your option) any later version.							#
#												#
#    This program is distributed in the hope that it will be useful,				#
#    but WITHOUT ANY WARRANTY; without even the implied warranty of				#
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the				#
#    GNU General Public License for more details.						#
#-----------------------------------------------------------------------------------------------#


#------------------------------- GPU and precision options for the force calculation
USING_CUDA = NO
#OPT += -DUSE_SINGLE_PRECISION

#------------------------------- Option for periodic boundary condition
#OPT += -DPERIODIC

#------------------------------- Option of HDF5
OPT += -DHAVE_HDF5

#------------------------------- Glass making option
#OPT += -DGLASS_MAKING

#------------------------------- Compiler
CXX = g++

#------------------------------- Location of the MPI libraries
MPI_LIBS = -L/usr/lib
MPI_INC = -I/usr/include/mpi

#------------------------------- Location of the CUDA Toolkit
CUDA_PATH       ?= /usr

#------------------------------- Location of the HDF5 libraries
HDF5_LIBS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
HDF5_INC = -I/usr/include/hdf5/serial 

#---------------- Do not edit beyond this line! ----------------

COMPILERVERSION=$(shell $(CXX) --version | head -1)
DATE=$(shell date +'%Y-%m-%d %A %H:%M %Z')
BUILD_NUMBER_LDFLAGS  = -DBUILD_DATE='"$(DATE)"' -DPROGRAM_VERSION=$(VERSION) -DCOMPILER_VERSION='"$(COMPILERVERSION)"'
CFLAGS = -Wno-long-long -Wno-unused-result -lm -Wall -ansi -pedantic -O3 -ffast-math -std=gnu++98 -fopenmp -lgomp

MPI_COMPILE_FLAGS = $(shell mpic++ --showme:compile)
MPI_LINK_FLAGS = $(shell mpic++ --showme:link)

ifeq ($(USING_CUDA), YES)
#------------------------------- Location of the CUDA Toolkit
OPT += -DUSE_CUDA
BUILD_NUMBER_LDFLAGS += -DPROGRAMNAME='"StePS_CUDA"'

NVCC = $(CUDA_PATH)/bin/nvcc -ccbin
CUDAFLAGS = -Xcompiler -fopenmp -lineinfo --compiler-options --std=gnu++98 --compiler-options -Wall --compiler-options -ansi -O3 -lm -Xcompiler -pthread -Xcompiler \'-Wl\\,-rpath\' -Xcompiler \'-Wl\\,$(MPI_LIBS)\' -Xcompiler \'-Wl\\,--enable-new-dtags\' -lmpi_cxx -lmpi
CUDA_INC = -I$(CUDA_PATH)/include

SRC = main.cc inputoutput.cc forces_cuda.cu ewald_space.cc step.cc read_paramfile.cc friedmann_solver.cc read_gadget_ic.cc
OBJ = main.o inputoutput.o forces_cuda.o ewald_space.o step.o read_paramfile.o friedmann_solver.o read_gadget_ic.o
DEPS = global_variables.h
PROG = StePS_CUDA

$(PROG): $(OBJ)
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ) $(HDF5_INC) $(HDF5_LIBS) $(MPI_LIBS)
%.o: %.cc
	$(CXX) $(CFLAGS) $(CUDA_INC) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(HDF5_LIBS) $(HDF5_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CXX) $(CUDAFLAGS) $(CUDALDFLAGS) $(CUDA_INC) $(MPI_LIBS) $(MPI_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<
clean:
	rm -fv $(OBJ) $(PROG)

else

BUILD_NUMBER_LDFLAGS += -DPROGRAMNAME='"StePS"'

SRC = main.cc inputoutput.cc forces.cc ewald_space.cc step.cc read_paramfile.cc friedmann_solver.cc read_gadget_ic.cc
OBJ = main.o inputoutput.o forces.o ewald_space.o step.o read_paramfile.o friedmann_solver.o read_gadget_ic.o
DEPS = global_variables.h
PROG = StePS
$(PROG): $(OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $(PROG) $(OBJ) $(HDF5_INC) $(HDF5_LIBS) $(MPI_LIBS)

%.o: %.cc
	$(CXX) $(CFLAGS) $(LDFLAGS) $(MPI_COMPILE_FLAGS) $(MPI_LINK_FLAGS) $(HDF5_LIBS) $(HDF5_INC) $(BUILD_NUMBER_LDFLAGS) $(OPT) -o $@ -c $<

clean:
	rm -fv $(OBJ) $(PROG)

endif
