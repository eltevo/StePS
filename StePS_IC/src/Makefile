
CXX = g++
BUILD_NUMBER_LDFLAGS  = -Xlinker --defsym -Xlinker __BUILD_DATE=$$(date +'%Y%m%d')
CFLAGS = -lm -Wno-long-long -Wall -ansi -pedantic -O3 -ffast-math -std=c++98 -fopenmp -lgsl -lgslcblas -lkdtree

#HEALpix cxx include directory
HEALPix_INC=-I${CURDIR}/../HEALPix/Healpix_3.31/src/C/autotools
HEALPix_LIB=-R${CURDIR}/../HEALPix/Healpix_3.31/src/C/autotools
#kdtree library directories
KDTREE_INC=-I/home/ragraat/Egyetem/kdtree-0.5.6
KDTREE_LIB=-L/home/ragraat/Egyetem/kdtree-0.5.6/build/lib -Wl,-R/home/ragraat/Egyetem/kdtree-0.5.6/build/lib


SRC = main.cc read_paramfile.cc read_gadget_ic.cc stereographic_projection.cc comoving_distance.cc
OBJ = main.o read_paramfile.o read_gadget_ic.o stereographic_projection.o comoving_distance.o
DEPS = global_variables.h
PROG = StePS_IC
$(PROG): $(OBJ)
	$(CXX) $(LDFLAGS) $(HEALPix_INC) $(KDTREE_INC) $(BUILD_NUMBER_LDFLAGS) -o $(PROG) $(OBJ) $(KDTREE_LIB) $(CFLAGS)

%.o: %.cc
	$(CXX) $(LDFLAGS) $(HEALPix_INC) $(KDTREE_INC) $(KDTREE_LIB) -o $@  $(CFLAGS) -c $< 


clean:
	rm -fv $(OBJ) $(PROG)

