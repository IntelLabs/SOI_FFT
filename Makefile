ifeq ($(MKLROOT),)
  #MKLROOT=/opt/intel/mkl/11.2.0.090/mkl
endif

CC=mpiicc

CFLAGS = -qopenmp -Qoption,cpp,--extended_float_type -Wall #-D_XOPEN_SOURCE=600
CFLAGS += -DMKL_ILP64 -I$(MKLROOT)/include -I$(BOOST_INCLUDE)
CFLAGS += -D_MPI -mt_mpi #-static_mpi
#DBG = yes
ifeq (yes, $(DBG))
  CFLAGS += -O0 -g
else
  CFLAGS += -O3 -xAVX -DNDEBUG
endif

MKL_LIB_DIR = $(MKLROOT)/lib/intel64
LDFLAGS = -L$(MKL_LIB_DIR) -Wl,--start-group $(MKL_LIB_DIR)/libmkl_cdft_core.a $(MKL_LIB_DIR)/libmkl_blacs_intelmpi_ilp64.a $(MKL_LIB_DIR)/libmkl_intel_ilp64.a $(MKL_LIB_DIR)/libmkl_intel_thread.a $(MKL_LIB_DIR)/libmkl_core.a -Wl,--end-group

ifeq (yes, $(FFTW))
  CFLAGS += -DSOI_USE_FFTW
  LDFLAGS += -lpthread -lfftw3_mpi -lfftw3_omp -lfftw3
endif

EXE_EXT=exe
OBJ_EXT=o
SRCS = pfft.c parallel_filter_subsampling.c parallel_filter_subsampling_n_mu_8.c input.c cpu_freq.c compress.c
TEST_SRCS = test.c $(SRCS)
OBJECTS = $(TEST_SRCS:.c=.$(OBJ_EXT))

all: test.exe

test_compress: compress.o
	$(CC) $(CFLAGS) $^ -o $@

input.$(OBJ_EXT): input.c
	mpiicpc -c $(CFLAGS) $< -o $@

%.$(OBJ_EXT): %.c
	$(CC) -c -std=c99 $(CFLAGS) $< -o $@

%.s: %.c
	$(CC) -c -std=c99 $(CFLAGS) $< -S -fsource-asm

%.$(EXE_EXT): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f test.$(EXE_EXT) test_compress.exe $(OBJECTS)

.PRECIOUS: test.o
