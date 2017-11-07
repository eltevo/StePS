
CXX = g++-4.8
BUILD_NUMBER_LDFLAGS  = -Xlinker --defsym -Xlinker __BUILD_DATE=$$(date +'%Y%m%d')
CFLAGS = -lm -Wall -ansi -pedantic -O3 -ffast-math -std=c++11 -fopenmp -lgsl -lgslcblas

#HEALpix cxx include directory
HEALPix_INC=-I${CURDIR}/../Healpix_3.31/src/C/autotools
HEALPix_LIB=-R${CURDIR}/../Healpix_3.31/src/C/autotools


SRC = main.cc read_paramfile.cc read_gadget_ic.cc stereographic_projection.cc comoving_distance.cc
OBJ = main.o read_paramfile.o read_gadget_ic.o stereographic_projection.o comoving_distance.o
DEPS = global_variables.h
PROG = StePS_IC
$(PROG): $(OBJ)
	$(CXX) $(LDFLAGS) $(HEALPix_INC) $(HEALPix_DIR) $(BUILD_NUMBER_LDFLAGS) -o $(PROG) $(OBJ) $(CFLAGS)

%.o: %.cc
	$(CXX) $(LDFLAGS) $(HEALPix_INC) $(HEALPix_DIR) -o $@  $(CFLAGS) -c $< 


clean:
	rm -fv $(OBJ) $(PROG)

