ifeq ($(MKLROOT),)
  #MKLROOT=/opt/intel/mkl/11.2.0.090/mkl
endif

CC=mpiicc

CFLAGS = -qopenmp -Qoption,cpp,--extended_float_type -D_XOPEN_SOURCE=600 #-vec-report=2
CFLAGS += -DMKL_ILP64 -I$(MKLROOT)/include # -I$(HOME)/usr/include
CFLAGS += -restrict -D_MPI -mt_mpi #-static_mpi
#DBG = yes
ifeq (yes, $(DBG))
  CFLAGS += -O0 -g -D_DEBUG
else
  CFLAGS += -O3 -xCORE-AVX2 -DNDEBUG
endif

MKL_LIB_DIR = $(MKLROOT)/lib/intel64
LDFLAGS = -L$(MKL_LIB_DIR) -Wl,--start-group $(MKL_LIB_DIR)/libmkl_cdft_core.a $(MKL_LIB_DIR)/libmkl_blacs_intelmpi_ilp64.a $(MKL_LIB_DIR)/libmkl_intel_ilp64.a $(MKL_LIB_DIR)/libmkl_intel_thread.a $(MKL_LIB_DIR)/libmkl_core.a -Wl,--end-group

ifeq (yes, $(FFTW))
  CFLAGS += -DUSE_FFTW
  LDFLAGS += -lpthread -lfftw3_mpi -lfftw3_omp -lfftw3
endif

EXE_EXT=exe
OBJ_EXT=o
SRCS = pfft.c parallel_filter_subsampling.c input.c cpu_freq.c compress.c
TEST_SRCS = test.c $(SRCS)
ACCURACY_SRCS = accuracy.c $(SRCS)
OBJECTS = $(TEST_SRCS:.c=.$(OBJ_EXT))
OBJECTS_ACCURACY = $(ACCURACY_SRCS:.c=.$(OBJ_EXT))
OBJECTS_SINGLE = $(SRCS:.c=.do)

all: test.exe

test_compress: compress.o
	$(CC) $(CFLAGS) $^ -o $@

input.$(OBJ_EXT): input.c
	mpiicpc -c $(CFLAGS) $< -o $@

%.$(OBJ_EXT): %.c
	$(CC) -c -std=c99 $(CFLAGS) $< -o $@

%.do: %.c
	$(CC) -c -std=c99 $(CFLAGS) $< -o $@ -DPRECISION=1

%.s: %.c
	$(CC) -c -std=c99 $(CFLAGS) $< -S -fsource-asm

%.$(EXE_EXT): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

test_single.exe: $(OBJECTS_SINGLE)
	$(CC) $(CFLAGS) $(OBJECTS_SINGLE) -o $@ $(LDFLAGS)

accuracy: $(OBJECTS_ACCURACY)
	$(CC) $(CFLAGS) $(OBJECTS_ACCURACY) -o $@ $(LDFLAGS)	

clean:
	rm -f test.$(EXE_EXT) test_compress.exe $(OBJECTS) $(OBJECTS_SINGLE)

.PRECIOUS: test.o
