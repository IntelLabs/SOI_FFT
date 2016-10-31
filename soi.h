#ifndef _SOI_FFT_SOI_H_
#define _SOI_FFT_SOI_H_

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stddef.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>

#include <complex.h>
#include "mpi.h"
#include <stdio.h>
#include <immintrin.h>
#include "mkl_dfti.h"

#ifdef SOI_USE_FFTW
#include <fftw3-mpi.h>
#endif

#include "intrinsic.h"

#define SOI_MEASURE_LOAD_IMBALANCE
#define SOI_USE_INTRINSIC
//#define SOI_USE_ALL_TO_ALL_SYNC
//#define SOI_USE_ALL_TO_ALL_ASYNC

#if PRECISION == 2
#define VAL_TYPE double
#define DFTI_TYPE DFTI_DOUBLE
#define MPI_TYPE MPI_DOUBLE

#ifdef SOI_USE_FFTW
#define FFTW_COMPLEX fftw_complex
#define FFTW_PLAN_WITH_NTHREADS fftw_plan_with_nthreads
#define FFTW_PLAN fftw_plan

#define FFTW_MPI_INIT() { fftw_init_threads(); fftw_mpi_init(); }
#define FFTW_MPI_LOCAL_SIZE_1D fftw_mpi_local_size_1d
#define FFTW_MALLOC fftw_malloc
#define FFTW_FREE fftw_free
#define FFTW_MPI_PLAN_DFT_1D fftw_mpi_plan_dft_1d
#define FFTW_MPI_EXECUTE_DFT fftw_mpi_execute_dft
#define FFTW_PLAN_DFT_1D fftw_plan_dft_1d
#define FFTW_EXECUTE fftw_execute
#define FFTW_EXECUTE_DFT fftw_execute_dft
#define FFTW_DESTROY_PLAN(a)
#define FFTW_CLEANUP_THREADS fftw_cleanup_threads
#define FFTW_MPI_CLEANUP fftw_mpi_cleanup
#endif

#else
// PRECISION == 1

#define VAL_TYPE float
#define DFTI_TYPE DFTI_SINGLE
#define MPI_TYPE MPI_FLOAT

#ifdef SOI_USE_FFTW
#define FFTW_COMPLEX fftwf_complex
#define FFTW_PLAN_WITH_NTHREADS fftwf_plan_with_nthreads
#define FFTW_PLAN fftwf_plan

#define FFTW_MPI_INIT() { fftwf_init_threads(); fftwf_mpi_init(); }
#define FFTW_MPI_LOCAL_SIZE_1D fftwf_mpi_local_size_1d
#define FFTW_MALLOC fftwf_malloc
#define FFTW_FREE fftwf_free
#define FFTW_MPI_PLAN_DFT_1D fftwf_mpi_plan_dft_1d
#define FFTW_MPI_EXECUTE_DFT fftwf_mpi_execute_dft
#define FFTW_PLAN_DFT_1D fftwf_plan_dft_1d
#define FFTW_EXECUTE fftwf_execute
#define FFTW_EXECUTE_DFT fftwf_execute_dft
#define FFTW_DESTROY_PLAN(a)
#define FFTW_CLEANUP_THREADS fftwf_cleanup_threads
#define FFTW_MPI_CLEANUP fftwf_mpi_cleanup
#endif

#endif // PRECISION == 1

#define CACHE_LINE_LEN (64/sizeof(VAL_TYPE))

#define TRANSPOSE(row0, row1, row2, row3) \
__m256d __t0, __t1, __t2, __t3; \
__t0 = _mm256_unpacklo_pd(row0, row1); \
__t1 = _mm256_unpackhi_pd(row0, row1); \
__t2 = _mm256_unpacklo_pd(row2, row3); \
__t3 = _mm256_unpackhi_pd(row2, row3); \
row0 = _mm256_permute2f128_pd(__t0, __t2, 0x20); \
row1 = _mm256_permute2f128_pd(__t1, __t3, 0x20); \
row2 = _mm256_permute2f128_pd(__t0, __t2, 0x31); \
row3 = _mm256_permute2f128_pd(__t1, __t3, 0x31);

//#define TRANSPOSE(row0, row1, row2, row3, row4, row5, row6, row7) \
__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7; \
__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7; \
__t0 = _mm256_unpacklo_ps(row0, row1); \
__t1 = _mm256_unpackhi_ps(row0, row1); \
__t2 = _mm256_unpacklo_ps(row2, row3); \
__t3 = _mm256_unpackhi_ps(row2, row3); \
__t4 = _mm256_unpacklo_ps(row4, row5); \
__t5 = _mm256_unpackhi_ps(row4, row5); \
__t6 = _mm256_unpacklo_ps(row6, row7); \
__t7 = _mm256_unpackhi_ps(row6, row7); \
__tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0)); \
__tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2)); \
__tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0)); \
__tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2)); \
__tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0)); \
__tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2)); \
__tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0)); \
__tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2)); \
row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20); \
row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20); \
row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20); \
row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20); \
row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31); \
row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31); \
row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31); \
row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);

#define TRANSPOSE_BLOCK_SIZE CACHE_LINE_LEN

#ifndef NDEBUG
#define DUMP(x) printf("%s\n", x)
#define DUMP_INT(x) printf("%s = %d\n", #x, x)
#define DUMP_FLOAT(x) printf("%s = %f\n", #x, x)
#define DUMP_PTR(x) printf("%s = %p\n", #x, x)
#define DUMP_CMPLX_ARRAY(x, sz) do {                                       \
	size_t __i;                                                            \
	complex_struct_t * __p = (complex_struct_t *)(x);                                          \
	for (__i=0; __i < (sz); __i++)                                         \
		printf("%s[%d] = %f + I*%f\n", #x, __i, __p[__i].re, __p[__i].im); \
} while (0)
#else
#define DUMP(x)
#define DUMP_INT(x)
#define DUMP_FLOAT(x)
#define DUMP_PTR(x)
#define DUMP_CMPLX_ARRAY(x, sz)
#endif

#define MPI_DUMP_INT(c, x)                 \
{                                          \
	int rank;                              \
	CFFT_ASSERT_MPI( MPI_Comm_rank(c, &rank) ); \
	if (rank == 0)                         \
		printf("%s\t%d\n", #x, (int)x);    \
}

#define MPI_DUMP_LONG(c, x)                 \
{                                          \
	int rank;                              \
	CFFT_ASSERT_MPI( MPI_Comm_rank(c, &rank) ); \
	if (rank == 0)                         \
		printf("%s\t%ld\n", #x, (long)x);    \
}

#define MPI_DUMP_FLOAT(c, x)               \
{                                          \
	int rank;                              \
	CFFT_ASSERT_MPI( MPI_Comm_rank(c, &rank) ); \
	if (rank == 0)                         \
		printf("%s\t%f\n", #x, x);         \
}

#define MPI_DUMP_1(c, s, x)                \
{                                          \
	int rank;                              \
	CFFT_ASSERT_MPI( MPI_Comm_rank(c, &rank) ); \
	if (rank == 0)                         \
		printf(s, #x, x);                  \
}

#define CFFT_ASSERT_MPI(f) \
do {                   \
long err;              \
err = f;               \
if (MPI_SUCCESS != err) \
  printf("MPI error %lu in %s (%s, line %lu)\n", (unsigned long)err, __FUNCTION__, __FILE__, (unsigned long)__LINE__); \
} while(0)

#define ASSERT_DFTI(x) if (DFTI_NO_ERROR != x) printf("Dfti error while %s\n", #x)

#define MPI_TIMED_SECTION_BEGIN() { double __timing = -MPI_Wtime();
#define MPI_TIMED_SECTION_END(c, x)            \
	{                                          \
		__timing += MPI_Wtime();               \
		if (d->rank == 0)                         \
			printf("%s\t%f\n", x, __timing);   \
	}                                          \
}
#define MPI_TIMED_SECTION_END_WO_NEWLINE(c, x)            \
	{                                          \
		__timing += MPI_Wtime();               \
		if (d->rank == 0)                         \
			printf("%s\t%f", x, __timing);   \
	}                                          \
}
#define MPI_TIMED_SECTION_END_WITH_BARRIER(c, x)            \
	{                                          \
		int rank;                              \
		CFFT_ASSERT_MPI( MPI_Barrier(c) );          \
		__timing += MPI_Wtime();               \
		CFFT_ASSERT_MPI( MPI_Comm_rank(c, &rank) ); \
		if (rank == 0)                         \
			printf("%s\t%f\n", x, __timing);   \
	}                                          \
}

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t cfft_size_t;

typedef struct _complex_struct_t {
	VAL_TYPE re;
	VAL_TYPE im;
} complex_struct_t;

typedef VAL_TYPE complex cfft_complex_t;
#define COMPLEX_PTR(x) ((cfft_complex_t *)(x))
#define MEMCPY_COMPLEX(x,y,n)                       \
do {                                                \
	cfft_size_t __i;                                \
	for (__i=0; __i<(n); __i++)                     \
		COMPLEX_PTR(x)[__i] = COMPLEX_PTR(y)[__i];  \
} while (0);

typedef struct
{
	MPI_Comm comm;
	int P;   // number of processors
	int rank; // procesor's id
	cfft_size_t k;   // number of segments per each processor
	cfft_size_t N;   // global vector length
	cfft_complex_t *W_inv; // inverse frequency window function tabulated values
	cfft_complex_t *w; // time window function tabulated values
  SIMDFPTYPE *w_dup;
	cfft_complex_t *gamma_tilde; // temp buf for sampled and filtered data of size M_hat*k
	cfft_complex_t *alpha_tilde; // another temp buf for permuted data of size M_hat*k
	cfft_complex_t *beta_tilde; // another temp buf for permuted data of size M_hat*k
  cfft_complex_t *alpha_ghost;
  int *delta, *epsilon;

	int n_mu;
	int d_mu; // d_mu should devide the input size
	cfft_size_t B;
  double tau; // a paramter controls the width of window function. The wider the width, the smaller truncation error becomes
  double sigma;

	DFTI_DESCRIPTOR_HANDLE desc_dft_s;
	DFTI_DESCRIPTOR_HANDLE desc_dft_m_hat;
#ifdef SOI_USE_FFTW
  int use_fftw;
  unsigned fftw_flags;
  FFTW_PLAN fftw_plan_s, fftw_plan_m_hat;
#endif
#if !defined(USE_ALL_TO_ALL_SYNC) && !defined(SOI_USE_ALL_TO_ALL_ASYNC)
  MPI_Request *sendRequests, *recvRequests;
#endif
#ifdef SOI_USE_ALL_TO_ALL_ASYNC
  pthread_t comm_thread;
  sem_t sem_recv;
#endif
  int use_vlc; // use variable length compression
  int *segmentBoundaries;
    // segmentBoundaries[i]: the first segment ith rank will process
  double comm_to_comp_cost_ratio;
} soi_desc_t;

__declspec(noinline)
void parallel_filter_subsampling(soi_desc_t *desc, cfft_complex_t * alpha_dt);

void set_default_soi_descriptor(soi_desc_t *desc);

void init_soi_descriptor(
  soi_desc_t *desc, MPI_Comm comm, cfft_size_t k, int use_fftw, unsigned fftw_flags);

void compute_soi(soi_desc_t * d, cfft_complex_t *alpha_dt);
void free_soi_descriptor(soi_desc_t * d);

void populate_input(cfft_complex_t *input, size_t localLen, size_t offset, size_t globalLen, int kind);
cfft_complex_t reference_output(size_t idx, size_t globalLen, int kind, size_t offset);
double compute_snr(
  cfft_complex_t *output, size_t localLen, size_t offset, size_t globalLen, int kind,
  soi_desc_t *d);
double compute_normalized_inf_norm(
  cfft_complex_t *output, size_t localLen, size_t offset, size_t globalLen, int kind);

__declspec (align(64)) static unsigned long Remaining[5][4] = {
  { 0x00, 0x00, 0x00, 0x00 },
  { 0xffffffffffffffffUL, 0x00, 0x00, 0x00 },
  { 0xffffffffffffffffUL, 0xffffffffffffffffUL, 0x00, 0x00 },
  { 0xffffffffffffffffUL, 0xffffffffffffffffUL, 0xffffffffffffffffUL, 0x00 },
  { 0xffffffffffffffffUL, 0xffffffffffffffffUL, 0xffffffffffffffffUL, 0xffffffffffffffffUL },
};
                                                            /*{  0,0,0,0,  0,0,0,0,   0,0,0,0,   0,0,0,0},
                                                            {  255,255,255,255,  0,0,0,0,   0,0,0,0,   0,0,0,0},
                                                            {  255,255,255,255,  255,255,255,255,   0,0,0,0,   0,0,0,0},
                                                            {  255,255,255,255,  255,255,255,255,   255,255,255,255,   0,0,0,0},
                                                            {  255,255,255,255,  255,255,255,255,   255,255,255,255,   255,255,255,255}};*/

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

double get_cpu_freq();

static const double PI=3.14159265358979323846;
#define VERIFY_PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086q
//static const double PI = 3.141592653589793115997963468544185161590576171875;

#define MAX_THREADS 512

#ifdef SOI_MEASURE_LOAD_IMBALANCE
extern unsigned long long load_imbalance_times[MAX_THREADS];

#ifdef __cplusplus
}
#endif

#endif

#endif // _SOI_FFT_SOI_H_
