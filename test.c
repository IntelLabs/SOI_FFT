#define _GNU_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <omp.h>

#include <mkl_cdft.h>

#include "soi.h"

/**
 * Command line options
 */
typedef struct options {
  int k_min, k_max;
  int input_min, input_max;
  int no_mkl, no_soi, no_snr;
  char *in_file_name;
  char *mkl_out_file_name;
  char *soi_out_file_name;
#ifdef SOI_USE_FFTW
  int no_fftw;
  char *fftw_out_file_name;
#endif
  int soi_with_fftw;
  unsigned fftw_flags;
} options;

static options parseArgs(int argc, char *argv[], soi_desc_t *desc)
{
  options ret;
  ret.k_min = ret.k_max = 8;
  ret.input_min = 2;
  ret.input_max = 2;
  ret.no_mkl = ret.no_soi = ret.no_snr = 0;
  ret.in_file_name = NULL;
  ret.mkl_out_file_name = NULL;
  ret.soi_out_file_name = NULL;
  ret.soi_with_fftw = 0;
#ifdef SOI_USE_FFTW
  ret.no_fftw = 0;
  ret.fftw_out_file_name = NULL;
  ret.fftw_flags = FFTW_ESTIMATE;
#endif

  while (1) {
    static struct option long_options[] = {
      { "k_min", required_argument, 0, 'k' },
      { "k_max", required_argument, 0, 'K' }, // sweep k over [k_min, k_max)
      { "input_min", required_argument, 0, 'i' },
      { "input_max", required_argument, 0, 'I' }, // sweep input kind over [input_min, input_max)
      { "no_mkl", no_argument, 0, 'o' },
      { "no_soi", no_argument, 0, 'O' },
      { "no_snr", no_argument, 0, 'c' }, // don't compute SNR
      { "n_mu", required_argument, 0, 'n' },
      { "d_mu", required_argument, 0, 'd' }, // over sampling factor = n_mu/d_mu . input size should be a multiple of d_mu
      { "tau", required_argument, 0, 'a' },
        // a parameter that controls the width of window function. The wider the width, the smaller truncation error becomes
      { "sigma", required_argument, 0, 'g' },
      { "B", required_argument, 0, 'B' },
      { "in_file", required_argument, 0, 'x' },
      { "mkl_out_file", required_argument, 0, 'm' },
      { "soi_out_file", required_argument, 0, 's' },
      { "vlc", no_argument, 0, 'v' },
      { "comm_to_comp_ratio", required_argument, 0, 'r' },
#ifdef SOI_USE_FFTW
      { "no_fftw", no_argument, 0, 'w' },
      { "fftw_out_file", required_argument, 0, 'f' },
      { "soi_with_fftw", no_argument, 0, 'F' },
      { "fftw_measure", no_argument, 0, 't' },
#endif
      { 0, 0, 0, 0 },
    };

    int option_index = 0;
    int c = getopt_long(argc, argv, "k:K:n:d:B:m:s:f:", long_options, &option_index);
    if (-1 == c) break;

    switch (c) {
    case 'k': ret.k_min = atoi(optarg); break;
    case 'K': ret.k_max = atoi(optarg); break;
    case 'i': ret.input_min = atoi(optarg); break;
    case 'I': ret.input_max = atoi(optarg); break;
    case 'o': ret.no_mkl = 1; break;
    case 'O': ret.no_soi = 1; break;
    case 'c': ret.no_snr = 1; break;

    case 'n': desc->n_mu = atoi(optarg); break;
    case 'd': desc->d_mu = atoi(optarg); break;
    case 'a': desc->tau = atof(optarg)/1024; break;
    case 'g': desc->sigma = atof(optarg); break;
    case 'B': desc->B = atoi(optarg); break;

    case 'x': ret.in_file_name = optarg; break;
    case 'm': ret.mkl_out_file_name = optarg; break;
    case 's': ret.soi_out_file_name = optarg; break;
    case 'v': desc->use_vlc = 1; break;
    case 'r': desc->comm_to_comp_cost_ratio = atof(optarg); break;
#ifdef SOI_USE_FFTW
    case 'w': ret.no_fftw = 1; break;
    case 'f': ret.fftw_out_file_name = optarg; break;
    case 'F': desc->use_fftw = 1; break;
    case 't': desc->fftw_flags = FFTW_MEASURE; break;
#endif
    case '?': break;
    default: exit(-1);
    }
  }

  if (optind >= argc) {
    fprintf(stderr, "usage: %s [k_min=k_min] [k_max=k_max] [n_mu=n_mu] [d_mu=d_mu] [B=B] [mkl_out_file=mkl_out_file] [soi_out_file=soi_out_file] [fftw_out_file=fftw_out_file] [vlc] N\n", argv[0]);
    exit(-1);
  }

  int powIdx = -1;
  for (int i = 0; i < strlen(argv[optind]); i++) {
    if (argv[optind][i] == '^') {
      powIdx = i;
      break;
    }
  }
  desc->N = (powIdx == -1) ? atol(argv[optind]) : pow(atol(argv[optind]), atol(argv[optind] + powIdx + 1));
  
  MPI_Comm_size(MPI_COMM_WORLD, &desc->P);
	MPI_Comm_rank(MPI_COMM_WORLD, &desc->rank);

  if (desc->N%(desc->d_mu*ret.k_max*desc->P*16) != 0) {
    if (0 == desc->rank) {
      fprintf(stderr, "(d_mu=%d)*(P=%d)*(k=%d)*64 must divide N\n", desc->d_mu, desc->P, ret.k_max);
    }
    exit(-1);
  }

  return ret;
}

static void initMPI(int argc, char *argv[])
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
}

/**
 * Each MPI rank sequentially writes its own buffer to a single file.
 */
static void mpiWriteFileSequentially(char *fileName, cfft_complex_t *buffer, size_t len)
{
  int P, rank;
  int ret, errLen;
  char buf[MPI_MAX_ERROR_STRING];

	ret = MPI_Comm_size(MPI_COMM_WORLD, &P);
  if (MPI_SUCCESS != ret) {
    MPI_Error_string(ret, buf, &errLen);
    fprintf(stderr, buf);
    exit(-1);
  }
	ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (MPI_SUCCESS != ret) {
    MPI_Error_string(ret, buf, &errLen);
    fprintf(stderr, buf);
    exit(-1);
  }

  for (int i = 0; i < P; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == i) {
      if (NULL != fileName) {
        FILE *fp = fopen(fileName, i == 0 ? "w" : "a");
        for (int i = 0; i < len; ++i) {
          fprintf(fp, "%g %g\n", __real__(buffer[i]), __imag__(buffer[i]));
        }
        fclose(fp);
      }
    }
  }
}

extern double time_begin_mpi, time_end_mpi;
extern double time_begin_fused[1024], time_end_fused[1024];

int main(int argc, char *argv[])
{
	soi_desc_t d;
  set_default_soi_descriptor(&d);

	cfft_complex_t *in_buf = NULL;
	double time_mkl, time_soi, max_err, g_max_err;
	DFTI_DESCRIPTOR_DM_HANDLE desc;

  initMPI(argc, argv);
  options options = parseArgs(argc, argv, &d);

  if (0 == d.rank) {
    printf(
      "P = %d, N = %ld, n_mu = %d, d_mu = %d, B = %ld, sigma = %f\n",
      d.P, d.N, d.n_mu, d.d_mu, d.B, d.sigma);
  }

  double flop = 5.*d.N*log2(d.N);

  const int REPEAT = 4;
  for (int iter = 0; iter < REPEAT; iter++) {
    /* for each input type */
    for (int input = options.input_min; input <= options.input_max; input++) {
      if (!options.no_mkl) {
        if (1 == d.P) { // 1 mpi rank : use local FFT
          DFTI_DESCRIPTOR_HANDLE desc;
          CHECK_DFTI( DftiCreateDescriptor(&desc, DFTI_TYPE, DFTI_COMPLEX, 1, d.N) );
          CHECK_DFTI( DftiCommitDescriptor(desc) );
          if (NULL == in_buf) {
            size_t in_buf_size = sizeof(cfft_complex_t)*d.N*d.n_mu/d.d_mu;
            posix_memalign((void **)&in_buf, 4096, in_buf_size);
            if (NULL == in_buf) {
              fprintf(stderr, "Failed to allocate in_buf (in_buf_size requested = %ld)\n", in_buf_size);
              return -1;
            }
          }
          populate_input(in_buf, d.N, 0, d.N, input);

          // Write input to file.
          mpiWriteFileSequentially(options.in_file_name, in_buf, d.N/d.P);

          time_mkl = -MPI_Wtime();
          CHECK_DFTI( DftiComputeForward(desc, in_buf) );
          time_mkl += MPI_Wtime();
          MPI_DUMP_FLOAT(MPI_COMM_WORLD, time_mkl);
          CHECK_DFTI( DftiFreeDescriptor(&desc) );
        }
        else {
          MKL_LONG status, size;
          CHECK_DFTI( DftiCreateDescriptorDM(MPI_COMM_WORLD, &desc, DFTI_TYPE, DFTI_COMPLEX, 1, d.N) );
          CHECK_DFTI( DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &size) );
          CHECK_DFTI( DftiSetValueDM(desc, DFTI_PLACEMENT, DFTI_INPLACE) );
          if (NULL == in_buf) {
            size_t in_buf_size = sizeof(cfft_complex_t)*size*d.n_mu/d.d_mu*2;
            posix_memalign((void **)&in_buf, 4096, in_buf_size);
            if (NULL == in_buf) {
              fprintf(stderr, "Failed to allocate in_buf (in_buf_size requested = %ld)\n", in_buf_size);
              return -1;
            }
          }
          CHECK_DFTI( DftiCommitDescriptorDM(desc) );
          populate_input(in_buf, d.N/d.P, d.rank*d.N/d.P, d.N, input);

          MPI_Barrier(MPI_COMM_WORLD);
          // Write input to file.
          mpiWriteFileSequentially(options.in_file_name, in_buf, d.N/d.P);

          MPI_Barrier(MPI_COMM_WORLD);
          time_mkl = -MPI_Wtime();
          CHECK_DFTI( DftiComputeForwardDM(desc, in_buf) );
          MPI_Barrier(MPI_COMM_WORLD);
          time_mkl += MPI_Wtime();
          MPI_DUMP_FLOAT(MPI_COMM_WORLD, time_mkl);
          CHECK_DFTI( DftiFreeDescriptorDM(&desc) );
        }

        if (0 == d.rank)
        {
          double gflops = flop/time_mkl/1e9;
          printf("flops_mkl\t%f\n", gflops);
        }

        // Write output to file.
        mpiWriteFileSequentially(options.mkl_out_file_name, in_buf, d.N/d.P);

        if (!options.no_snr) {
          double mkl_snr = INFINITY, mkl_max_err = 0;
          if (input != 2 && input != 4 && input != 5) {
            mkl_snr = compute_snr(in_buf, d.N/d.P, d.rank*d.N/d.P, d.N, input, NULL);
            mkl_max_err = compute_normalized_inf_norm(in_buf, d.N/d.P, d.rank*d.N/d.P, d.N, input);
          }
          if (0 == d.rank) {
            printf("snr_mkl%d\t%f\n", input, mkl_snr);
            printf("max_err_mkl%d\t%e\n", input, mkl_max_err);
          }
        }
      }

#ifdef SOI_USE_FFTW
      if (!options.no_fftw) {
        // FFTW
        
        double time_fftw;
        ptrdiff_t local_ni = N, local_i_start = 0, local_no = N, local_o_start = 0;
        FFTW_PLAN fftw_plan;

        if (1 == d.P) {
          FFTW_PLAN_WITH_NTHREADS(omp_get_max_threads());

          if (in_buf == NULL) {
            in_buf = (FFTW_COMPLEX *)FFTW_MALLOC(N*sizeof(FFTW_COMPLEX)*n_mu/d_mu);
            if (NULL == in_buf) {
              fprintf(stderr, "Failed to allocated local data\n");
              return -1;
            }
          }
          fftw_plan = FFTW_PLAN_DFT_1D(
            N, in_buf, in_buf, FFTW_FORWARD, options.fftw_flags);

          populate_input(in_buf, N, 0, N, input);

          time_fftw = -MPI_Wtime();
          FFTW_EXECUTE(fftw_plan);
          time_fftw += MPI_Wtime();
        }
        else {
          FFTW_MPI_INIT();
          FFTW_PLAN_WITH_NTHREADS(omp_get_max_threads());
          ptrdiff_t total_local_size = FFTW_MPI_LOCAL_SIZE_1D(
            N, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE,
            &local_ni, &local_i_start, &local_no, &local_o_start);

          if (in_buf == NULL) {
            in_buf = (FFTW_COMPLEX *)FFTW_MALLOC(total_local_size*sizeof(FFTW_COMPLEX)*n_mu/d_mu);
            if (NULL == in_buf) {
              fprintf(stderr, "Failed to allocated local data\n");
              return -1;
            }
          }

          fftw_plan = FFTW_MPI_PLAN_DFT_1D(
            N, in_buf, in_buf, MPI_COMM_WORLD, FFTW_FORWARD, options.fftw_flags);

          populate_input(in_buf, local_ni, local_i_start, N, input);

          time_fftw = -MPI_Wtime();
          FFTW_MPI_EXECUTE_DFT(fftw_plan, in_buf, in_buf);
          //FFTW_EXECUTE(fftw_plan);
          time_fftw += MPI_Wtime();
        }

        MPI_DUMP_FLOAT(MPI_COMM_WORLD, time_fftw);

        FFTW_DESTROY_PLAN(fftw_plan);

        // Write output to file.
        mpiWriteFileSequentially(options.fftw_out_file_name, in_buf, local_no);

        if (!options.no_snr) {
          double fftw_snr = compute_snr(in_buf, local_ni, local_i_start, N, input, NULL);
          double fftw_max_err = compute_normalized_inf_norm(in_buf, local_ni, local_i_start, N, input);
          if (0 == d.rank) {
            printf("snr_fftw%d\t%f\n", input, fftw_snr);
            printf("max_err_fftw%d\t%e\n", input, fftw_max_err);
          }
        }

        FFTW_CLEANUP_THREADS();
        if (d.P > 1) FFTW_MPI_CLEANUP();
      }
#endif

      //////////////////////////////
      for (int soi_with_fftw = 0; soi_with_fftw <= options.soi_with_fftw; ++soi_with_fftw) {
        for (int k = options.k_min; k <= options.k_max && !options.no_soi; k *= 2) {
          init_soi_descriptor(&d, MPI_COMM_WORLD, k);

          cfft_size_t S = d.k*d.P; // total number of segments
          cfft_size_t M = d.N/S; // length of one segment, before oversampling
          cfft_size_t M_hat = d.n_mu*M/d.d_mu; // length of one segment, after oversampling

          if (in_buf == NULL) {
            size_t in_buf_size = sizeof(cfft_complex_t)*M_hat*d.k;
            posix_memalign((void **)&in_buf, 4096, in_buf_size);
            if (NULL == in_buf) {
              fprintf(stderr, "Failed to allocate in_buf (in_buf_size requested = %ld)\n", in_buf_size);
              return -1;
            }
          }

          populate_input(in_buf, d.N/d.P, d.rank*d.N/d.P, d.N, input);
          MPI_Barrier(MPI_COMM_WORLD);
          time_soi = -MPI_Wtime();
          compute_soi(&d, in_buf);

          MPI_Barrier(MPI_COMM_WORLD);
          time_soi += MPI_Wtime();
          if (0 == d.rank) {
            if (soi_with_fftw) {
              printf("time_soi_fftw_%d\t%f\n", k, time_soi);
            }
            else {
              printf("time_soi_%d\t%f\n", k, time_soi);
            }
            double gflops = flop/time_soi/1e9;
            printf("flops_soi_%d\t%f\n", k, gflops);
          }

          // even though these buffers will be deallocated in free_soi_descriptor,
          // we deallocate them earlier here so that we can have enough memory
          // required to compute SNR
          free(d.w); d.w = NULL;
          free(d.w_dup); d.w_dup = NULL;
          free(d.W_inv); d.W_inv = NULL;
          free(d.alpha_ghost); d.alpha_ghost = NULL;
          if (d.use_vlc && d.epsilon) {
            free(d.epsilon); d.epsilon = NULL;
          }

          // don't free the following buffers to reuse across multiple SOI FFT runs
          d.alpha_tilde = NULL;
          d.beta_tilde = NULL;
          d.gamma_tilde = NULL;

          if (!options.no_snr) {
            int firstSegment = d.segmentBoundaries[d.rank];
            int nSegments = d.segmentBoundaries[d.rank + 1] - firstSegment;

            double soi_snr = compute_snr(
              in_buf, M*nSegments, M*firstSegment, d.N, input, &d);
            double soi_max_err = compute_normalized_inf_norm(
              in_buf, M*nSegments, M*firstSegment, d.N, input);
            if (0 == d.rank) {
              if (soi_with_fftw) {
                printf("snr_soi_fftw%d_%d\t%f\n", input, k, soi_snr);
                printf("max_err_soi_fftw%d_%d\t%e\n", input, k, soi_max_err);
              }
              else {
                printf("snr_soi%d_%d\t%f\n", input, k, soi_snr);
                printf("max_err_soi%d_%d\t%e\n", input, k, soi_max_err);
              }
            }
          }

#ifdef SOI_PRINT_MPI_TIMES
          int numOfSegToReceive =
            d.segmentBoundaries[d.rank + 1] - d.segmentBoundaries[d.rank];

          for (int p = 0; p < d.P; ++p) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (d.rank == p) {
              FILE *fp = fopen("time.out", p == 0 ? "w" : "a");
              fprintf(fp, "[%d] %f %f ", p, time_begin_mpi, time_end_mpi);
              for (int ik = 0; ik < numOfSegToReceive; ++ik) {
                fprintf(fp, "%f %f ", time_begin_fused[ik], time_end_fused[ik]);
              }
              fprintf(fp, "\n");
              fclose(fp);
            }
          } // for each rank
#endif

          free_soi_descriptor(&d);

          // Write output to file.
          mpiWriteFileSequentially(options.soi_out_file_name, in_buf, d.N/d.P);
        } // for (int k = kmin; k <= kmax; k *= 2)
      }
    } // for (int input = 0; input < 2; input++)
  }

	MPI_Finalize();	

  return 0;
}
