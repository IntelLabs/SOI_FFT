#define _GNU_SOURCE
#include <sched.h>
#include <limits.h>
#include <assert.h>
#include "soi.h"
#include <mkl_trans.h>
#include <mkl_types.h>
#include <float.h>
#include <omp.h>
#include <mkl_service.h>
#include "compress.h"

/*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parallel 1-D FFT based on Segment of Interest algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--A high level description
%The result space is divided in S segment of interest. Usually S is a
%multiple of the number of processor P. Thus, if N is the original data
%length, each segment is of length M = N/S. 
%Denote the (global) input data by "alpha", a vector of lenght N,
%the main idea of the algorithm is that corresponding to each segment s, 
%s = 0, 1, ..., S-1, of the result, we construct a condensed input vector by 
%filtering and subsampling the input "alpha" to give alpha_tilde^(s) 
%of length approximately M. The DFT of this condensed input yields the
%resulting segment after simple demodulation. The exact length of these
%alpha_tilde^(s) is M_hat, which is longer than M. This is due to the
%nature of this algorithm. Oversampling is needed to maintain numerical
%stability and accuracy. M_hat = mu * M, mu is an oversampling factor
%mu is of the form mu = n_mu / d_mu. We typically try n_mu = d_mu + 1 and
%d_mu is 2, 4, or 8.
%
%The whole algorithm has 3 major steps:
%1. Filter and Subsampling (algorithmically the most sophisticated part)
%2. Straightforward DFT of length M_hat
%3. Division by W, which is done by multiplication by 1/W.
%

%We mimic a cluster implementation here. But to aid testing, we maintain
%the "global" version of key data, such as the complete input and the
%complete "alpha_tilde" as well as "gamma_tilde"
%We named the gloabal and distributed data with an obvious "glb" and "dt"
%suffixes.
%
*/

// The best one found after solving the issue with SNR measurement.
static cfft_size_t kappa;

cfft_complex_t w_f(cfft_size_t i, cfft_size_t n, soi_desc_t *desc)
{
  cfft_size_t S = desc->k*desc->P; // total number of segments
	cfft_size_t M = desc->N/S;
	cfft_size_t M_hat = desc->n_mu*M/desc->d_mu;

  cfft_size_t theta = i/(desc->B*S);
  cfft_size_t j = i%(desc->B*S);

  double t = ((double)theta/M_hat - (double)j/n + (double)kappa/(2*M))*M;
  double y;

  if (t == 0) {
    y = 1;
  }
  else {
    y = sinl(VERIFY_PI*desc->tau*t)/(PI*desc->tau*t)*exp(-PI*PI*t*t/desc->sigma);
  }

  cfft_complex_t r = cosl(VERIFY_PI*t) + I*sinl(VERIFY_PI*t);

  double mu = (double)(desc->n_mu)/desc->d_mu;
  return r*y/mu;
}

cfft_complex_t W_inv_f(cfft_size_t i, cfft_size_t n, soi_desc_t *desc)
{
  cfft_size_t S = desc->k*desc->P; // total number of segments
	cfft_size_t M = desc->N/S;

  double t = (double)i/M - 0.5;

  double y = 1/(2*desc->tau)*(erfc(sqrt(desc->sigma)*(t - desc->tau/2)) - erfc(sqrt(desc->sigma)*(t + desc->tau/2)));

  double delta = (double)kappa / (2*M);
  cfft_complex_t r = cosl(2*VERIFY_PI*delta*i) - I*sinl(2*VERIFY_PI*delta*i);

  return r/y;
}

static cfft_complex_t *g_gamma_tilde= NULL, *g_alpha_tilde = NULL, *g_beta_tilde = NULL;

void init_soi_descriptor(
  soi_desc_t *d, MPI_Comm comm, cfft_size_t k, int use_fftw, unsigned fftw_flags)
{
	int rank, size;

  kappa = d->B - d->d_mu;

	d->comm = comm;
	d->k = k;
  d->segmentBoundaries = (int *)malloc(sizeof(int)*(d->P + 1));
  for (int p = 0; p <= d->P; ++p) {
    d->segmentBoundaries[p] = p*k;
  }
  cfft_size_t S = d->k*d->P; // total number of segments
	cfft_size_t M = d->N/S;
	cfft_size_t M_hat = d->n_mu*M/d->d_mu;
  posix_memalign((void **)&d->w, 4096, sizeof(cfft_complex_t)*d->B*S*d->n_mu);
  posix_memalign((void **)&d->w_dup, 4096, sizeof(SIMDFPTYPE)*d->B*S*d->n_mu);
  posix_memalign((void **)&d->W_inv, 4096, sizeof(cfft_complex_t)*M);
  if (NULL == g_gamma_tilde) {
    posix_memalign((void **)&g_gamma_tilde, 4096, sizeof(cfft_complex_t)*M_hat*k*2);
  }
	d->gamma_tilde = g_gamma_tilde;
  if (NULL == d->gamma_tilde) {
    fprintf(stderr, "Failed to allocate d->gamma_tilde\n");
    exit(1);
  }

  if (NULL == g_alpha_tilde) {
    posix_memalign((void **)&g_alpha_tilde, 4096, sizeof(cfft_complex_t)*M_hat*k);
  }
	d->alpha_tilde = g_alpha_tilde;
  if (NULL == d->alpha_tilde) {
    fprintf(stderr, "Failed to allocate d->alpha_tilde\n");
    exit(1);
  }

  if (NULL == g_beta_tilde) {
    posix_memalign((void **)&g_beta_tilde, 4096, sizeof(cfft_complex_t)*M_hat*k);
  }
  d->beta_tilde = g_beta_tilde;
  if (NULL == d->beta_tilde) {
    fprintf(stderr, "Failed to allocate d->beta_tilde\n");
    exit(1);
  }

  //d->alpha_ghost = (cfft_complex_t *)_mm_malloc(sizeof(cfft_complex_t)*d->M*S, 4096);
  posix_memalign((void **)&d->alpha_ghost, 4096, sizeof(cfft_complex_t)*2*d->B*S);
  if (NULL == d->alpha_ghost) {
    fprintf(stderr, "Failed to allocate d->alpha_ghost\n");
    exit(1);
  }
  posix_memalign((void **)&d->delta, 4096, sizeof(int)*4*M_hat*k);
  d->epsilon = NULL;

  d->sendRequests = (MPI_Request *)malloc(sizeof(MPI_Request)*d->P*d->k);
  d->recvRequests = (MPI_Request *)malloc(sizeof(MPI_Request)*d->P*S);

	for (int theta=0; theta<d->n_mu; theta++)
#pragma omp parallel for
    for (cfft_size_t i = 0; i < S/(SIMD_WIDTH/2)*(SIMD_WIDTH/2); i += CACHE_LINE_LEN/2) {
      for (cfft_size_t j = 0; j < d->B; j++)
        for (cfft_size_t ii = i; ii < i + CACHE_LINE_LEN/2; ii++)
          (d->w)[i*d->B*d->n_mu + (j*d->n_mu + theta)*(CACHE_LINE_LEN/2) + ii - i] =
            w_f(theta*d->B*S + j*S + ii, d->N, d);

      for (cfft_size_t j = 0; j < d->B; j++) {
        for (cfft_size_t ii = 0; ii < 2; ii++) {
          SIMDFPTYPE temp = _MM_LOADU((VAL_TYPE *)(d->w + i*d->B*d->n_mu + (j*d->n_mu + theta)*(CACHE_LINE_LEN/2)) + ii*SIMD_WIDTH);
          _MM_STOREU(d->w_dup + i*d->B*d->n_mu + (j*d->n_mu + theta)*(CACHE_LINE_LEN/2) + 2*ii, _MM_MOVELDUP(temp));
          _MM_STOREU(d->w_dup + i*d->B*d->n_mu + (j*d->n_mu + theta)*(CACHE_LINE_LEN/2) + 2*ii + 1, _MM_MOVEHDUP(temp));
        }
      }
    }
		/*for (cfft_size_t i=0; i<d->B; i++) {
			for (cfft_size_t j=0; j<S; j++)
				(d->w)[theta*d->B*S + i*S + j] = (d->w_f)(theta*d->B*S + i*S + j, d->N);

      for (cfft_size_t j = 0; j < S/(SIMD_WIDTH/2)*(SIMD_WIDTH/2); j += SIMD_WIDTH/2) {
        SIMDFPTYPE temp = _MM_LOADU((VAL_TYPE *)(d->w + theta*d->B*S + i*S + j));
        _MM_STOREU(d->w_dup + (theta*d->B*S + i*S + j)*SIMD_WIDTH, _MM_MOVELDUP(temp));
        _MM_STOREU(d->w_dup + (theta*d->B*S + i*S + j + 1)*SIMD_WIDTH, _MM_MOVEHDUP(temp));
      }
    }*/

#pragma omp parallel for
	for (cfft_size_t i=0; i < M; i++) {
		(d->W_inv)[i] = W_inv_f(i, d->N, d);
  }

	// create DFT descriptors
#ifdef USE_FFTW
  d->use_fftw = use_fftw;
  d->fftw_flags = fftw_flags;
  if (use_fftw) {
    FFTW_PLAN_WITH_NTHREADS(omp_get_max_threads());
    d->fftw_plan_s = FFTW_PLAN_DFT_1D(
      S, d->gamma_tilde, d->gamma_tilde, FFTW_FORWARD, fftw_flags);
    d->fftw_plan_m_hat = FFTW_PLAN_DFT_1D(
      M_hat, d->alpha_tilde, d->alpha_tilde, FFTW_FORWARD, fftw_flags);
  }
  else
#endif
  {
    ASSERT_DFTI( DftiCreateDescriptor(&(d->desc_dft_s), DFTI_TYPE, DFTI_COMPLEX, 1, (long)S) );
    ASSERT_DFTI( DftiSetValue(d->desc_dft_s, DFTI_NUMBER_OF_USER_THREADS, omp_get_max_threads()) );
    //ASSERT_DFTI( DftiSetValue(d->desc_dft_s, DFTI_NUMBER_OF_TRANSFORMS, 8) );
    //ASSERT_DFTI( DftiSetValue(d->desc_dft_s, DFTI_INPUT_DISTANCE, S) );
    //ASSERT_DFTI( DftiSetValue(d->desc_dft_s, DFTI_OUTPUT_DISTANCE, S) );
    ASSERT_DFTI( DftiCommitDescriptor(d->desc_dft_s) );
    ASSERT_DFTI( DftiCreateDescriptor(&(d->desc_dft_m_hat), DFTI_TYPE, DFTI_COMPLEX, 1, (long)(M_hat)) );
    MKL_LONG status = DftiCommitDescriptor(d->desc_dft_m_hat);
    if (status & !DftiErrorClass(status, DFTI_NO_ERROR))
      fprintf(stderr, "%s\n", DftiErrorMessage(status));
  }

  if (0 == d->rank) printf("freq = %f\n", get_cpu_freq());
}

void free_soi_descriptor(soi_desc_t * d)
{
	// free DFT descriptors
#ifdef USE_FFTW
  if (d->use_fftw) {
    FFTW_DESTROY_PLAN(d->fftw_plan_s);
    FFTW_DESTROY_PLAN(d->fftw_plan_m_hat);
    FFTW_CLEANUP_THREADS();
  }
  else
#endif
  {
    ASSERT_DFTI( DftiFreeDescriptor(&(d->desc_dft_s)) );
    ASSERT_DFTI( DftiFreeDescriptor(&(d->desc_dft_m_hat)) );
  }

  // free requests
  //CFFT_ASSERT_MPI(MPI_Waitall(d->P*d->k, d->sendRequests, MPI_STATUSES_IGNORE));
  free(d->sendRequests);
  free(d->recvRequests); d->recvRequests = NULL;

	// free window functions tables
	if (d->w) free(d->w);
	if (d->w_dup) free(d->w_dup);
	if (d->W_inv) free(d->W_inv);
	//free(d->alpha_tilde);
	if (d->alpha_ghost) free(d->alpha_ghost);
  //if (d->gamma_tilde) free(d->gamma_tilde); d->gamma_tilde = NULL;
  if (d->use_vlc) {
    if (d->epsilon) free(d->epsilon); d->epsilon = NULL;
  }
  if (d->segmentBoundaries) free(d->segmentBoundaries); d->segmentBoundaries = NULL;
}

#ifdef MEASURE_LOAD_IMBALANCE
unsigned long long load_imbalance_times[MAX_THREADS];
#endif

double time_begin_mpi, time_end_mpi;
double time_begin_fused[1024], time_end_fused[1024];

void mpiWriteFileSequentially(char *fileName, cfft_complex_t *buffer, size_t len);

void compute_soi(soi_desc_t * d, cfft_complex_t *alpha_dt)
{
  double soiBeginTime = MPI_Wtime();

  cfft_size_t S = d->k*d->P; // total number of segments
  cfft_size_t M = d->N/S; // length of one segment, before oversampling
  cfft_size_t M_hat = d->n_mu*M/d->d_mu; // length of one segment, after oversampling
	cfft_size_t l = M_hat/d->P;

/*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 1: parallel filter and subsampling
%         The result of this step is that gamma_tilde is produced
%         Each processor does filtering and subsampling using mostly
%         its own values of   alpha_dt( one's own processor, : )
%         But it does require some alpha_dt of its next processor.
%         We use the name   gamma_tilde_dt  for the distributed gamma_tilde
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/
MPI_TIMED_SECTION_BEGIN();
	parallel_filter_subsampling(d, alpha_dt);
#ifdef MEASURE_LOAD_IMBALANCE
MPI_TIMED_SECTION_END_WO_NEWLINE(d->comm, "time_fss_total");
  if (0 == d->rank) {
    double min = DBL_MAX, max = DBL_MIN;
    for (int i = 0; i < omp_get_max_threads(); i++) {
      double t = load_imbalance_times[i]/get_cpu_freq();
      min = MIN(min, t); max = MAX(max, t);
    }
    printf(" load_imbalance = %f~%f\n", min, max);
  }
#else
MPI_TIMED_SECTION_END(d->comm, "time_fss_total");
#endif

  int sendLens[S];
  int totalMaxExponent;
  int *globalMaxExponentReduced;
  double time_compress = 0;

  if (d->use_vlc) {
    // Compute the maximum exponent of each segment
    double ttt = -MPI_Wtime();
    int *maxExponent = (int *)malloc(sizeof(int)*S);
    for (int s = 0; s < S; ++s) {
      maxExponent[s] = max_exponent((double *)(d->alpha_tilde + s*l), l*2);
    }
    ttt += MPI_Wtime();
    if (0 == d->rank) {
      printf("compute maximum exp takes %f\n", ttt);
    }
    time_compress += ttt;

    globalMaxExponentReduced = (int *)malloc(sizeof(int)*S);
    ttt = -MPI_Wtime();
    CFFT_ASSERT_MPI(MPI_Allreduce(
      maxExponent, globalMaxExponentReduced, S,
      MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    ttt += MPI_Wtime();
    if (0 == d->rank) {
      printf("MPI_Allreduce takes %f\n", ttt);
    }
    time_compress += ttt;

    totalMaxExponent = INT_MIN;
    for (int s = 0; s < S; ++s) {
      totalMaxExponent = MAX(totalMaxExponent, globalMaxExponentReduced[s]);
    }

    if (0 == d->rank) {
      for (int s = 0; s < S; ++s) {
        printf("%d ", globalMaxExponentReduced[s]);
      }
      printf("\n%d\n", totalMaxExponent);
      printf("l = %d\n", l);
    }

  } // use_vlc

  /*// Compute the maximum magnitude of each segment
  double *maxMagnitude = (double *)malloc(sizeof(double)*S);
  for (int s = 0; s < S; ++s) {
    maxMagnitude[s] = 0;
    for (int i = 0; i < l; ++i) {
      cfft_complex_t c = d->alpha_tilde[s*l + i];
      maxMagnitude[s] = MAX(maxMagnitude[s], fabs(creal(c)));
      maxMagnitude[s] = MAX(maxMagnitude[s], fabs(cimag(c)));
    }
  }

  double *globalMaxMagnitudeReduced = (double *)malloc(sizeof(double)*S);
  MPI_Allreduce(
    maxMagnitude, globalMaxMagnitudeReduced, S,
    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double totalMax = 0;
  for (int s = 0; s < S; ++s) {
    totalMax = MAX(totalMax, globalMaxMagnitudeReduced[s]);
  }

  if (0 == d->rank) {
    for (int s = 0; s < S; ++s) {
      printf("%g ", globalMaxMagnitudeReduced[s]);
    }
    printf("\n%g\n", totalMax);
  }*/

  double time_fused_mpi = 0, time_fused_fft = 0, time_fused_vmul = 0;
  double time_decompress = 0;
  double temp_time, time_fused;

  int sCnts[S];
  int sDispls[S], rDispls[S];

  if (d->use_vlc) {
    double partialCostSum[S + 1];
    partialCostSum[0] = 0;

    for (int p = 0; p < d->P; ++p) {
      for (int ik = 0; ik < d->k; ++ik) {
        int s = p*d->k + ik;
        int bitsPerInput = 52 + 3 - (totalMaxExponent - globalMaxExponentReduced[s]);
        if (bitsPerInput > 31) ++bitsPerInput;
        int lenPerSimdLane = (l*2/16*bitsPerInput + 31)/32;
        int remainder = (l*2%16*bitsPerInput + 31)/32;
        int sendLen = lenPerSimdLane*16 + remainder;
        if (bitsPerInput <= 0) sendLen = 0;

        sCnts[s] = (sendLen + 1)/2;

        partialCostSum[s + 1] =
          partialCostSum[s] +
          d->comm_to_comp_cost_ratio*sCnts[s]/2/M_hat + 1;
      }
    }

    // load balance
    double costPerRank = partialCostSum[S]/d->P;
    int s = 1;
    for (int p = 1; p < d->P; ) {
      if (partialCostSum[s] >= p*costPerRank) {
        if (partialCostSum[s] - p*costPerRank >
            p*costPerRank - partialCostSum[s - 1]) {
          d->segmentBoundaries[p] = s - 1;
        }
        else {
          d->segmentBoundaries[p] = s;
        }
        ++p;
      }
      else {
        ++s;
      }
    }

    if (0 == d->rank) {
      for (int p = 0; p < d->P; ++p) {
        printf(
          "%d: %g %d\n",
          p,
          partialCostSum[d->segmentBoundaries[p + 1]] - partialCostSum[d->segmentBoundaries[p]],
          d->segmentBoundaries[p + 1] - d->segmentBoundaries[p]);
      }
    }
  } // d->use_vlc
  else {
    for (int i = 0; i < d->P; ++i) {
      sCnts[i] = 2*l; // *2 for sizeof(complex double)/sizeof(double)
      sDispls[i] = i*2*d->k*l;
      rDispls[i] = i*2*l;
    }
  }

  int numOfSegToReceive =
    d->segmentBoundaries[d->rank + 1] - d->segmentBoundaries[d->rank];

  /*if (g_gamma_tilde) free(g_gamma_tilde);
  g_gamma_tilde = (cfft_complex_t *)_mm_malloc(
    sizeof(cfft_complex_t)*d->M_hat*numOfSegToReceive,
    4096);
  d->gamma_tilde = g_gamma_tilde;*/

  if (d->use_vlc) {
    if (d->epsilon) free(d->epsilon);
    posix_memalign((void **)&d->epsilon, 4096, sizeof(int)*4*M_hat*numOfSegToReceive);
  }

  long compressedLen = 0;
  double time_mpi = 0;

  int maxNSegment = 0;
  for (int p = 0; p < d->P; ++p) {
    maxNSegment = MAX(
      d->segmentBoundaries[p + 1] - d->segmentBoundaries[p], maxNSegment);
  }

  time_begin_mpi = MPI_Wtime() - soiBeginTime;

  for (cfft_size_t ik = 0; ik < MAX(maxNSegment, numOfSegToReceive); ik++) {
    if (d->use_vlc) {
      time_mpi -= MPI_Wtime();
      // pairwise exchange algorithm
      if (ik < numOfSegToReceive) {
        for (int i = 0; i < d->P; ++i) {
          int src = (d->rank - i + d->P)%d->P;
          int segment = d->segmentBoundaries[d->rank] + ik;

          CFFT_ASSERT_MPI(MPI_Irecv(
            d->epsilon + (ik*d->P + src)*l*4,
            sCnts[segment],
            MPI_TYPE, src, segment,
            d->comm, d->recvRequests + ik*d->P + src));
        } // for each MPI rank
      }
      time_mpi += MPI_Wtime();

      if (ik < maxNSegment) {
        time_compress -= MPI_Wtime();
#pragma omp parallel for
        for (int p = 0; p < d->P; ++p) {
          int segment = d->segmentBoundaries[p + 1] - maxNSegment + ik;
          if (segment < d->segmentBoundaries[p]) continue;
            // FIXME: load balancing

          unsigned long long t = __rdtsc();
          sendLens[segment] = compress(
            d->delta + segment*l*4, (double *)(d->alpha_tilde + segment*l),
            l*2,
            totalMaxExponent, globalMaxExponentReduced[segment],
            0/*7 == d->rank && 3 == s*/);

          compressedLen += sendLens[segment];
          assert(sCnts[segment] == (sendLens[segment] + 1)/2);

          //memcpy(d->epsilon, d->alpha_tilde + s*l, sizeof(double)*l*2);

          /*t = __rdtsc();
          decompress(
            (double *)(d->alpha_tilde + s*l), d->delta + s*l*4,
            l*2,
            totalMaxExponent, globalMaxExponentReduced[s],
            (double *)d->epsilon);
          if (0 == s) {
            printf("decompress takes %f\n", (__rdtsc() - t)/get_cpu_freq());
          }*/
        }

        time_compress += MPI_Wtime();

        time_mpi -= MPI_Wtime();

        // pairwise exchange algorithm
        for (int i = 0; i < d->P; ++i) {
          int dst = (d->rank + i)%d->P;

          int segment = d->segmentBoundaries[dst + 1] - maxNSegment + ik;
          if (segment < d->segmentBoundaries[dst]) continue;

          CFFT_ASSERT_MPI(MPI_Isend(
            d->delta + segment*l*4, sCnts[segment],
            MPI_TYPE, dst, segment,
            d->comm, d->sendRequests + segment));
        }

        time_mpi += MPI_Wtime();
      }
    } // d->use_vlc
    else {
      time_mpi -= MPI_Wtime();

#ifdef USE_I_ALL_TO_ALL
      CFFT_ASSERT_MPI(MPI_Ialltoallv(
        d->alpha_tilde + ik*l, sCnts, sDispls, MPI_TYPE,
        d->gamma_tilde + ik*d->P*l, sCnts, rDispls, MPI_TYPE,
        d->comm, d->recvRequests + ik));
#else
      // pairwise exchange algorithm
      if (ik < numOfSegToReceive) {
        for (int i = 0; i < d->P; ++i) {
          int src = (d->rank - i + d->P)%d->P;
          int segment = d->segmentBoundaries[d->rank] + ik;

          CFFT_ASSERT_MPI(MPI_Irecv(
            d->gamma_tilde + (ik*d->P + src)*l, l*2,
            MPI_TYPE, src, segment,
            d->comm, d->recvRequests + ik*d->P + src));
        } // for each MPI rank
      }

      if (ik < maxNSegment) {
        for (int i = 0; i < d->P; ++i) {
          int dst = (d->rank + i)%d->P;

          int segment = d->segmentBoundaries[dst + 1] - maxNSegment + ik;
          if (segment < d->segmentBoundaries[dst]) continue;

          CFFT_ASSERT_MPI(MPI_Isend(
            d->alpha_tilde + segment*l, l*2,
            MPI_TYPE, dst, segment,
            d->comm, d->sendRequests + segment));
        }
      }
#endif

      time_mpi += MPI_Wtime();
    } // !d->use_vlc
  }
  if (0 == d->rank) {
    if (d->use_vlc) {
      printf("compression rate = %g\n", (double)compressedLen/(l*S*4));
      printf("time_compress\t%f\n", time_compress);
    }
    printf("time_mpi\t%f\n", time_mpi);
  }

  time_fused = MPI_Wtime();
  time_end_mpi = time_fused - soiBeginTime;

#ifdef MEASURE_LOAD_IMBALANCE
  for (int i = 0; i < omp_get_max_threads(); i++)
    load_imbalance_times[i] = 0;
#endif

	for (cfft_size_t ik = 0; ik < numOfSegToReceive; ik++)
	{
    temp_time = MPI_Wtime();
#ifdef USE_I_ALL_TO_ALL
    assert(!d->use_vlc); // i_all_to_all doesn't work with vlc
    CFFT_ASSERT_MPI(MPI_Wait(d->recvRequests + ik, MPI_STATUS_IGNORE));
#else
    CFFT_ASSERT_MPI(MPI_Waitall(
      d->P, d->recvRequests + ik*d->P, MPI_STATUSES_IGNORE));
#endif
    temp_time = MPI_Wtime() - temp_time;
    //if (0 == d->rank) printf("\ttime_fused_mpi = %f", temp_time);
    time_fused_mpi += temp_time;
    temp_time = MPI_Wtime();
    //if (0 == d->rank) printf("%s:%d\n", __FILE__, __LINE__);

    time_begin_fused[ik] = temp_time - soiBeginTime;

    if (d->use_vlc) {
      double t_decompress = MPI_Wtime();
#pragma omp parallel for
      for (int p = 0; p < d->P; ++p) {
        int *srcBuffer = d->epsilon + (ik*M_hat + p*l)*4;
        cfft_complex_t *dstBuffer = d->gamma_tilde + ik*M_hat + p*l;
        decompress(
          (double *)dstBuffer, srcBuffer,
          l*2,
          totalMaxExponent,
          globalMaxExponentReduced[d->segmentBoundaries[d->rank] + ik],
          NULL);
      }
      time_decompress += MPI_Wtime() - t_decompress;
    }

    /*if (3 == ik) {
      mpiWriteFileSequentially(
        "segment_3.out",
        d->gamma_tilde + ik*M_hat,
        M_hat);
    }*/

    temp_time = MPI_Wtime();
#ifdef USE_FFTW
    if (d->use_fftw)
      FFTW_EXECUTE_DFT(
        d->fftw_plan_m_hat, d->gamma_tilde + ik*M_hat, d->gamma_tilde + ik*M_hat);
    else {
#endif
      {
		    DftiComputeForward(d->desc_dft_m_hat, d->gamma_tilde + ik*M_hat);
      }

    double t2 = MPI_Wtime();
    //if (0 == d->rank) printf("\ttime_fused_fft = %f\n", t2 - temp_time);
    time_fused_fft += t2 - temp_time;

#if defined(INTRINSIC)
#pragma omp parallel
    {
#pragma omp for
    for (cfft_size_t i = 0; i < M/(SIMD_WIDTH/2)*(SIMD_WIDTH/2); i += SIMD_WIDTH/2) {
      SIMDFPTYPE xtemp = _MM_LOAD((VAL_TYPE *)(d->W_inv + i));
      SIMDFPTYPE xl = _MM_MOVELDUP(xtemp);
      SIMDFPTYPE xh = _MM_MOVEHDUP(xtemp);
      SIMDFPTYPE ytemp = _MM_LOAD((VAL_TYPE *)(d->gamma_tilde + ik*M_hat + i));
      SIMDFPTYPE temp = _MM_FMADDSUB(xl, ytemp, _MM_SWAP_REAL_IMAG(_MM_MUL(xh, ytemp)));
      _MM_STREAM((VAL_TYPE *)(alpha_dt + ik*M + i), temp);
    }

#ifdef MEASURE_LOAD_IMBALANCE
    unsigned long long t = __rdtsc();
#pragma omp barrier
    load_imbalance_times[omp_get_thread_num()] += __rdtsc() - t;
#endif
    }
    cfft_size_t i = M/(SIMD_WIDTH/2)*(SIMD_WIDTH/2);

    if (i < M) {
      SIMDFPTYPE xtemp = _MM_LOADU((VAL_TYPE *)(d->W_inv + i));
      SIMDFPTYPE xl = _MM_MOVELDUP(xtemp);
      SIMDFPTYPE xh = _MM_MOVEHDUP(xtemp);
      SIMDFPTYPE ytemp = _MM_LOADU((VAL_TYPE *)(d->gamma_tilde + ik*M_hat + i));
      __m256i mask = _mm256_load_si256((__m256i *)Remaining[M - i]);
      SIMDFPTYPE temp = _MM_FMADDSUB(xl, ytemp, _MM_SWAP_REAL_IMAG(_MM_MUL(xh, ytemp)));
      _MM_MASKSTORE((VAL_TYPE *)(alpha_dt + ik*M + i), mask, temp);
    }
#else
#pragma omp parallel
    {
#pragma omp for
#pragma simd
    for (cfft_size_t i = 0; i < M; i++)
      alpha_dt[ik*M + i] = d->W_inv[i]*d->gamma_tilde[ik*M_hat + i];

#ifdef MEASURE_LOAD_IMBALANCE
    unsigned long long t = __rdtsc();
#pragma omp barrier
    load_imbalance_times[omp_get_thread_num()] += __rdtsc() - t;
#endif
    }
#endif
    time_fused_vmul += MPI_Wtime() - t2;

    time_end_fused[ik] = MPI_Wtime() - soiBeginTime;
	} // for each segment

/*#ifdef MEASURE_LOAD_IMBALANCE
  if (0 == d->rank) {
    printf("time_fused\t%f", MPI_Wtime() - time_fused);
    double min = DBL_MAX, max = DBL_MIN;
    for (int i = 0; i < omp_get_max_threads(); i++) {
      double t = load_imbalance_times[i]/get_cpu_freq();
      min = MIN(min, t); max = MAX(max, t);
    }
    printf(" load_imbalance = %f~%f\n", min, max);
  }
#else
MPI_TIMED_SECTION_END(d->comm, "time_fused");
#endif*/
  if (0 == d->rank) {
    printf("time_fused\t%f\n", MPI_Wtime() - time_fused);
    printf(
      "\ttime_fused_mpi = %f\ttime_decompress = %f\ttime_fused_fft = %f\ttime_fused_vmul = %f\n",
      time_fused_mpi, time_decompress, time_fused_fft, time_fused_vmul);
  }
#ifndef USE_I_ALL_TO_ALL
  CFFT_ASSERT_MPI(MPI_Waitall(d->P*d->k, d->sendRequests, MPI_STATUSES_IGNORE));
#endif
  //omp_set_num_threads(omp_get_max_threads() + 1);
  //printf("%d\n", omp_get_max_threads());
}
