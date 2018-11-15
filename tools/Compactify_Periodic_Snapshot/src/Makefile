#--------------------------------------------------------------C++ compiler
CXX = g++
#--------------------------------------------------------------HEALpix cxx include directory
HEALPix_INC=-I${CURDIR}/../HEALPix/Healpix_3.31/src/C/autotools
HEALPix_LIB=-R${CURDIR}/../HEALPix/Healpix_3.31/src/C/autotools
#--------------------------------------------------------------kdtree library directories
KDTREE_INC=-I${CURDIR}/../kdtree-0.5.6/build/include
KDTREE_LIB=-L${CURDIR}/../kdtree-0.5.6/build/lib -Wl,-R/home/username/pathto/kdtree-0.5.6/build/lib



#---------------- Do not edit beyond this line! ----------------
COMPILERVERSION=$(shell $(CXX) --version | head -1)
DATE=$(shell date +'%Y-%m-%d %A %H:%M %Z')
BUILD_NUMBER_LDFLAGS  = -DBUILD_DATE='"$(DATE)"' -DCOMPILER_VERSION='"$(COMPILERVERSION)"'
CFLAGS = -lm -Wno-long-long -Wno-unused-result -Wall -ansi -pedantic -O3 -ffast-math -std=c++98 -fopenmp -lgsl -lgslcblas -lkdtree

SRC = main.cc read_paramfile.cc read_gadget_ic.cc stereographic_projection.cc comoving_distance.cc
OBJ = main.o read_paramfile.o read_gadget_ic.o stereographic_projection.o comoving_distance.o
DEPS = global_variables.h
PROG = Compactify_Periodic_Snapshot
$(PROG): $(OBJ)
	$(CXX) $(LDFLAGS) $(HEALPix_INC) $(KDTREE_INC) $(BUILD_NUMBER_LDFLAGS) -o $(PROG) $(OBJ) $(KDTREE_LIB) $(CFLAGS)

%.o: %.cc
	$(CXX) $(LDFLAGS) $(HEALPix_INC) $(KDTREE_INC) $(KDTREE_LIB) $(BUILD_NUMBER_LDFLAGS) -o $@  $(CFLAGS) -c $< 


clean:
	rm -fv $(OBJ) $(PROG)

