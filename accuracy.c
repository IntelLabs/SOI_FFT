#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <getopt.h>
#include <mkl_cdft.h>

#include "soi.h"

cfft_size_t N;
cfft_size_t kmin, kmax;
cfft_size_t n_mu = 5;
cfft_size_t d_mu = 4;

extern double tau;
extern double sigma;
extern cfft_size_t B;

cfft_complex_t w_f(cfft_size_t i, cfft_size_t n);
cfft_complex_t W_inv_f(cfft_size_t i, cfft_size_t n);

typedef struct options {
  int k_min, k_max;
  int input_min, input_max;
  int no_mkl, no_soi, no_snr;
  char *mkl_out_file_name;
  char *soi_out_file_name;
#ifdef USE_FFTW
  int no_fftw;
  char *fftw_out_file_name;
#endif
  int soi_with_fftw;
  unsigned fftw_flags;
  int iteration;
} options;

static options parseArgs(int argc, char *argv[])
{
  options ret;
  ret.k_min = ret.k_max = 8;
  ret.input_min = 0;
  ret.input_max = 1;
  ret.no_mkl = ret.no_soi = ret.no_snr = 0;
  ret.mkl_out_file_name = NULL;
  ret.soi_out_file_name = NULL;
  ret.soi_with_fftw = 0;
  ret.iteration = 1;
#ifdef USE_FFTW
  ret.no_fftw = 0;
  ret.fftw_out_file_name = NULL;
  ret.fftw_flags = FFTW_ESTIMATE;
#endif

  while (1) {
    static struct option long_options[] = {
      { "k_min", required_argument, 0, 'k' },
      { "k_max", required_argument, 0, 'K' },
      { "input_min", required_argument, 0, 'i' },
      { "input_max", required_argument, 0, 'I' },
      { "no_mkl", no_argument, 0, 'o' },
      { "no_soi", no_argument, 0, 'O' },
      { "no_snr", no_argument, 0, 'c' },
      { "n_mu", required_argument, 0, 'n' },
      { "d_mu", required_argument, 0, 'd' },
      { "tau", required_argument, 0, 'a' },
      { "sigma", required_argument, 0, 'g' },
      { "B", required_argument, 0, 'B' },
      { "mkl_out_file", required_argument, 0, 'm' },
      { "soi_out_file", required_argument, 0, 's' },
      { "iteration", required_argument, 0, 'N' },
#ifdef USE_FFTW
      { "no_fftw", no_argument, 0, 'w' },
      { "fftw_out_file", required_argument, 0, 'f' },
      { "soi_with_fftw", no_argument, 0, 'F' },
      { "fftw_measure", no_argument, 0, 't' },
#endif
      { 0, 0, 0, 0 },
    };

    int option_index = 0;
    int c = getopt_long(argc, argv, "k:K:n:d:B:m:s:f:N:", long_options, &option_index);
    if (-1 == c) break;

    switch (c) {
    case 'k': ret.k_min = atoi(optarg); break;
    case 'K': ret.k_max = atoi(optarg); break;
    case 'i': ret.input_min = atoi(optarg); break;
    case 'I': ret.input_max = atoi(optarg); break;
    case 'o': ret.no_mkl = 1; break;
    case 'O': ret.no_soi = 1; break;
    case 'c': ret.no_snr = 1; break;
    case 'n': n_mu = atoi(optarg); break;
    case 'd': d_mu = atoi(optarg); break;
    case 'a': tau = atof(optarg)/1024; break;
    case 'g': sigma = atof(optarg); break;
    case 'B': B = atoi(optarg); break;
    case 'm': ret.mkl_out_file_name = optarg; break;
    case 's': ret.soi_out_file_name = optarg; break;
    case 'N': ret.iteration = atoi(optarg); break;
#ifdef USE_FFTW
    case 'w': ret.no_fftw = 1; break;
    case 'f': ret.fftw_out_file_name = optarg; break;
    case 'F': ret.soi_with_fftw = 1; break;
    case 't': ret.fftw_flags = FFTW_MEASURE; break;
#endif
    case '?': break;
    default: exit(-1);
    }
  }

  if (optind >= argc) {
    fprintf(stderr, "usage: %s [k_min=k_min] [k_max=k_max] [n_mu=n_mu] [d_mu=d_mu] [B=B] [mkl_out_file=mkl_out_file] [soi_out_file=soi_out_file] [fftw_out_file=fftw_out_file] N\n", argv[0]);
    exit(-1);
  }

  int powIdx = -1;
  for (int i = 0; i < strlen(argv[optind]); i++) {
    if (argv[optind][i] == '^') {
      powIdx = i;
      break;
    }
  }
  N = (powIdx == -1) ? atol(argv[optind]) : pow(atol(argv[optind]), atol(argv[optind] + powIdx + 1));

  return ret;
}

static void initMPI(int argc, char *argv[], int *P, int *PID)
{
  int ret, len;
  char buf[MPI_MAX_ERROR_STRING];

  int provided;
	ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (MPI_SUCCESS != ret) {
    MPI_Error_string(ret, buf, &len);
    fprintf(stderr, buf);
    exit(-1);
  }
  if (provided < MPI_THREAD_SERIALIZED) {
    fprintf(stderr, "MPI doesn't provide MPI_THREAD_SERIALIZED\n");
    exit(-1);
  }
	ret = MPI_Comm_size(MPI_COMM_WORLD, P);
  if (MPI_SUCCESS != ret) {
    MPI_Error_string(ret, buf, &len);
    fprintf(stderr, buf);
    exit(-1);
  }
	ret = MPI_Comm_rank(MPI_COMM_WORLD, PID);
  if (MPI_SUCCESS != ret) {
    MPI_Error_string(ret, buf, &len);
    fprintf(stderr, buf);
    exit(-1);
  }
}

double computeSnr(
  cfft_complex_t *outBuf, cfft_complex_t *refBuf, size_t len) {
  double powerErr = 0, powerSig = 0;
#pragma omp parallel for reduction(+:powerErr, powerSig)
  for (size_t idx = 0; idx < len; idx++) {
    cfft_complex_t ref = refBuf[idx];
    cfft_complex_t diff = outBuf[idx] - ref;
    powerErr += creal(diff)*creal(diff) + cimag(diff)*cimag(diff);
    powerSig += creal(ref)*creal(ref) + cimag(ref)*cimag(ref);
  }

  double globalPowerErr, globalPowerSig;
  MPI_Reduce(&powerErr, &globalPowerErr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&powerSig, &globalPowerSig, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return 10*log10(globalPowerSig/globalPowerErr);
}

int main(int argc, char *argv[])
{
  soi_desc_t *soiDesc;
  int P, PID;
  initMPI(argc, argv, &P, &PID);
  options options = parseArgs(argc, argv);
  
  DFTI_DESCRIPTOR_HANDLE mklDesc;
  DFTI_DESCRIPTOR_DM_HANDLE mklDmDesc;

#ifdef USE_FFTW
  FFTW_PLAN fftwPlan;
#endif

  cfft_complex_t *mklBuf =
    (cfft_complex_t *)_mm_malloc(sizeof(cfft_complex_t)*N/P*n_mu/d_mu, 4096);
  if (NULL == mklBuf) {
    fprintf(stderr, "Failed to allocate mklBuf\n");
    return -1;
  }
  cfft_complex_t *fftwBuf =
    (cfft_complex_t *)_mm_malloc(sizeof(cfft_complex_t)*N/P*n_mu/d_mu, 4096);
  if (NULL == fftwBuf) {
    fprintf(stderr, "Failed to allocate fftwBuf\n");
    return -1;
  }
  cfft_complex_t *soiBuf =
    (cfft_complex_t *)_mm_malloc(sizeof(cfft_complex_t)*N/P*n_mu/d_mu, 4096);
  if (NULL == soiBuf) {
    fprintf(stderr, "Failed to allocate soiBuf\n");
    return -1;
  }

  if (!options.no_mkl) {
    if (1 == P) {
      DftiCreateDescriptor(&mklDesc, DFTI_TYPE, DFTI_COMPLEX, 1, N);
      DftiCommitDescriptor(mklDesc);
    }
    else {
      DftiCreateDescriptorDM(
        MPI_COMM_WORLD, &mklDmDesc, DFTI_TYPE, DFTI_COMPLEX, 1, N);
      DftiSetValueDM(mklDmDesc, DFTI_PLACEMENT, DFTI_INPLACE);
      DftiCommitDescriptorDM(mklDmDesc);
    }
  }

#ifdef USE_FFTW
  if (!options.no_fftw) {
    if (1 == P) {
      FFTW_PLAN_WITH_NTHREADS(omp_get_max_threads());
      fftwPlan = FFTW_PLAN_DFT_1D(
        N, fftwBuf, fftwBuf, FFTW_FORWARD, options.fftw_flags);
    }
    else {
      FFTW_MPI_INIT();
      FFTW_PLAN_WITH_NTHREADS(omp_get_max_threads());
      ptrdiff_t local_ni = N, local_i_start = 0, local_no = N, local_o_start = 0;
      FFTW_MPI_LOCAL_SIZE_1D(
        N, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE,
        &local_ni, &local_i_start, &local_no, &local_o_start);
      fftwPlan = FFTW_MPI_PLAN_DFT_1D(
        N, fftwBuf, fftwBuf, MPI_COMM_WORLD, FFTW_FORWARD, options.fftw_flags);
    }
  }
#endif

  int k = 8;
  double mklSnrMin = 65535, mklSnrMax = 0, mklSnrSum = 0;
#ifdef USE_FFTW
  double fftwSnrMin = 65535, fftwSnrMax = 0, fftwSnrSum = 0;
#endif
  for (int i = 0; i < options.iteration; i++) {
    for (int i = 0; i < N/P; i++) {
      mklBuf[i] = (double)rand()/RAND_MAX + I*(double)rand()/RAND_MAX;
      fftwBuf[i] = mklBuf[i];
      soiBuf[i] = mklBuf[i];
    }

    create_soi_descriptor(
      &soiDesc, MPI_COMM_WORLD, N, k, n_mu, d_mu, w_f, W_inv_f, B,
      0, options.fftw_flags);

    compute_soi(soiDesc, soiBuf);

    free_soi_descriptor(soiDesc);

    if (!options.no_mkl) {
      if (1 == P) DftiComputeForward(mklDesc, mklBuf);
      else DftiComputeForwardDM(mklDmDesc, mklBuf);

      double snr = computeSnr(soiBuf, mklBuf, N/P);
      mklSnrMin = MIN(snr, mklSnrMin);
      mklSnrMax = MAX(snr, mklSnrMax);
      mklSnrSum += snr;
      if (0 == PID) printf("mkl %lf\n", snr);
    }

#ifdef USE_FFTW
    if (!options.no_fftw) {
      if (1 == P) FFTW_EXECUTE(fftwPlan);
      else FFTW_MPI_EXECUTE_DFT(fftwPlan, fftwBuf, fftwBuf);

      double snr = computeSnr(soiBuf, fftwBuf, N/P);
      fftwSnrMin = MIN(snr, fftwSnrMin);
      fftwSnrMax = MAX(snr, fftwSnrMax);
      fftwSnrSum += snr;
      if (0 == PID) printf("fftw %lf\n", snr);
    }
#endif
  }

  if (0 == PID) {
    printf(
      "mkl avg %lf min %lf max %lf\n",
      mklSnrSum/options.iteration, mklSnrMin, mklSnrMax);
#ifdef USE_FFTW
    printf(
      "fftw avg %lf min %lf max %lf\n",
      fftwSnrSum/options.iteration, fftwSnrMin, fftwSnrMax);
#endif
  }

  if (!options.no_mkl) {
    if (1 == P) DftiFreeDescriptor(&mklDesc);
    else DftiFreeDescriptorDM(&mklDmDesc);
  }

#ifdef USE_FFTW
  if (!options.no_fftw) {
    FFTW_DESTROY_PLAN(fftw_plan);
    FFTW_CLEANUP_THREADS();
    if (P > 1) FFTW_MPI_CLEANUP();
  }
#endif
}
