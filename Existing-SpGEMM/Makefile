CXX = icc

CFLAGS = -qopenmp -std=c++17 
#CFLAGS += -DKNL_EXE

#CFLAGS += -g
CFLAGS += -O2

#CFLAGS += -DDEBUG

#CFLAGS += -DCPP
CFLAGS += -DTBB

LDFLAGS = -L "/opt/intel/oneapi/mkl/latest/lib/intel64" -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm
LDFLAGS += -ltbbmalloc


BIN = ./bin
SRC = ./src
OBJ = ./obj
INC = ./inc

INCLUDE = -I/opt/intel/oneapi/tbb/latest/include/tbb
INCLUDE += -I$(INC)

OBJLIB = $(OBJ)/Timings.o

$(OBJ)/%.o : $(SRC)/%.cpp 
	mkdir -p $(dir $@)
	$(CXX) -c $(CFLAGS) $(INCLUDE) -o $@ $<

heap_spgemm : $(OBJ)/heap_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

hash_spgemm : $(OBJ)/hash_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

hashvec_spgemm : $(OBJ)/hashvec_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

outer_spgemm : $(OBJ)/outer_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

mkl_spgemm : $(OBJ)/mkl_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)


reg_heap_spgemm : $(OBJ)/reg_heap_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

reg_hash_spgemm : $(OBJ)/reg_hash_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

reg_hashvec_spgemm : $(OBJ)/reg_hashvec_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

reg_outer_spgemm : $(OBJ)/reg_outer_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

reg_mkl_spgemm : $(OBJ)/reg_mkl_spgemm.o $(OBJLIB)
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)



clean :
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
