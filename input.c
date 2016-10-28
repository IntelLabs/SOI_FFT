#include <assert.h>
#include <math.h>

#include "soi.h"
#include <mkl.h>
#include <mkl_cdft.h>
#include <omp.h>
#include <fstream>

#define VERIFY_TYPE double

#define COS cosl
#define SIN sinl

using namespace std;

extern "C" {

static const VERIFY_TYPE theta = 1;
static const VERIFY_TYPE relativeW = 1./16;
static const size_t invRelativeW = 16;

void populate_input(
  cfft_complex_t *input, size_t localLen, size_t offset, size_t globalLen, int kind) {
  if (5 == kind) { // digital spotlighiting in UHPC SAR benchmark
    FILE *fp = fopen("../data/sar_fft.in", "r");
    cfft_complex_t *temp = (cfft_complex_t *)malloc(sizeof(cfft_complex_t)*64000);

    fread(temp, sizeof(cfft_complex_t), 64000, fp);

#pragma omp parallel for
    for (int i = 0; i < localLen; ++i) {
      if (offset + i >= 64000) {
        input[i] = 0;
      }
      else {
        input[i] = temp[offset + i];
      }
    }

    free(temp);
    fclose(fp);
  }
  else if (4 == kind) { // astronomy data from Colfax

    const double A1 = 1.050780e-02;
    const double PB = 6.511572e-02;
    const double T0 = 5.539053e+04;
    const double epoch = 300000000.0;

    const char *fileName = "colfax/FT1_J1311_simple.dat";

    //printf("Processing %s with A1=%e PB=%e T0=%e\n", fileName, A1, PB, T0);

    // Probing the input file
    FILE* inf=fopen(fileName, "r");
    if (!inf) {
      printf("Failed to open file %s\n", fileName);
      exit(1);
    }
    char str[100];
    int nPhotons=0;
    while (!feof(inf)) {
      fgets(str, 100, inf);
      nPhotons++;
    }
    nPhotons -= (4+1);
    //printf("nPhotons=%d\n", nPhotons);

    // Reading the input file
    // Skip 4-line header
    fseek(inf, 0, SEEK_SET);
    for (long i = 0; i < 4; i++) 
      fgets(str, 100, inf);
    // and read photon arrival times and weights
    double arrivalTime[nPhotons], weight[nPhotons];
    for (long i = 0; i < nPhotons; i++) {
      fscanf(inf, "%lE %lE", &arrivalTime[i], &weight[i]);
      arrivalTime[i] -= epoch;
    }
    fclose(inf);

    // Preparing the Fourier transform engine
    // In production code this is done once for processing multiple parameters,
    // so the duration of this operation is not important.
    const double maxF0=2048.0f; // Max spin frequency in Hz
    const double minF0=256.0f;  // Min spin frequency in Hz
    const double maxDiff=524288.0; // Max difference in photon arrival times, sec
    const long fftSize = 1L<<31L; // Equal to 2*maxDiff*maxF0
    //printf("Preparing FFT engine... ");
    double t0 = omp_get_wtime();
#pragma omp parallel for
    for (long i = 0; i < localLen; i++)
      input[i] = 0.0f;
    double t1 = omp_get_wtime();
    //printf("%.6f seconds\n", t1-t0);

    // The following operations in production code are performed in a loop
    // over multiple values of search parameters A1, PB and T0

    // Demodulating photon arrival times
    //printf("Demodulating photon arrival times... ");
    t0 = omp_get_wtime();
    const double O_Hz = 3.141592653589793238462643/(43200.0 * PB); // Orbital frequency in Hz
    const double T0_MET = (T0 - 51910.0) * 86400.0 - 64.184; // T0 in MET (Mission Elapsed Time)
    for (long i = 0; i < nPhotons; i++)
      arrivalTime[i] -= A1*sin(O_Hz*(arrivalTime[i] + epoch - T0_MET)); 
    t1 = omp_get_wtime();
    //printf("%.6f seconds\n", t1-t0);

    // Computing differences in photon arrival times
    // and hashing these time differences into the FFT array
    //printf("Computing time differences and building FFT data... ");
    t0 = omp_get_wtime();
    double totalWeight = 0.0;
    long nDiffs = 0;
#pragma omp parallel for reduction(+: nDiffs) reduction(+: totalWeight)
    for (long i = 0; i < nPhotons-1; i++)
      // Only computing positive differences
      for (long j = i+1; j < nPhotons; j++) {
        const double timeDiff = arrivalTime[j] - arrivalTime[i];

        // We need to discuss with the author whether 
        // these breaks are correct. At worst,
        // they decrease the search sensitivity.
        if (timeDiff<0) break;
        if (timeDiff>maxDiff) break;

        // This check is correct
        if ((0.0 < timeDiff) && (timeDiff < maxDiff)) {
          const double weightDiff=weight[j]*weight[i];
          totalWeight += weightDiff;

          const long tBin = long(timeDiff * 2.0 * double(maxF0) + 0.5);

          if (tBin >= offset && tBin < offset + localLen) {
#pragma omp atomic
            input[tBin - offset] += weightDiff;
          }

          nDiffs++;
        }
      }
    t1 = omp_get_wtime();
    //printf("%.6f seconds\n", t1-t0);
  }
  else if (2 == kind) { // for compression test
    const double SIGMA = .3; //1.;
    const double LENGTH = 80;

#pragma omp parallel for
    for (size_t idx = 0; idx < localLen; ++idx) {
      input[idx] = 0;
    }

#include "dhfr_particles.c"

#pragma omp parallel
    {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    const size_t BUF_LEN = 1024;
    double buf[BUF_LEN], buf2[BUF_LEN];

    size_t iPerThread = ((localLen + BUF_LEN - 1)/BUF_LEN + nthreads - 1)/nthreads*BUF_LEN;
    size_t iBegin = min(tid*iPerThread, localLen);
    size_t iEnd  = min(iBegin + iPerThread, localLen);

    const size_t NUM_LINES_PER_DIM = 1;
    size_t samplesPerLine =
      (globalLen + NUM_LINES_PER_DIM*NUM_LINES_PER_DIM - 1)/(NUM_LINES_PER_DIM*NUM_LINES_PER_DIM);
    size_t sampleLen = 3584;

    for (int i = 0; i < sizeof(dhfr_particles)/sizeof(dhfr_particles[0]); ++i) {
      double charge = dhfr_particles[i][0];
      double x = dhfr_particles[i][1];
      double y = dhfr_particles[i][2];
      double z = dhfr_particles[i][3];

      for (size_t idx = iBegin; idx < iEnd; idx += BUF_LEN) {
        if (idx + offset >= sampleLen) continue;
        for (int idx2 = idx; idx2 < min(idx + BUF_LEN, iEnd); ++idx2) {
          int ix = (idx2 + offset)%samplesPerLine;
          int iy = (idx2 + offset)/samplesPerLine%NUM_LINES_PER_DIM;
          int iz = (idx2 + offset)/samplesPerLine/NUM_LINES_PER_DIM;

          double x0 = LENGTH*ix/sampleLen - LENGTH/2;
          double y0 = 1 == NUM_LINES_PER_DIM ? 0 : LENGTH*iy/NUM_LINES_PER_DIM - LENGTH/2;
          double z0 = 1 == NUM_LINES_PER_DIM ? 0 : LENGTH*iz/NUM_LINES_PER_DIM - LENGTH/2;

          double r2 = (x - x0)*(x - x0) + (y - y0)*(y - y0) + (z - z0)*(z - z0);
          buf[idx2 - idx] = -r2/SIGMA;
          //input[idx2] += charge*exp(-r2/SIGMA);
        }
        vmdExp(min(BUF_LEN, iEnd - idx), buf, buf2, VML_EP);
        for (int idx2 = idx; idx2 < min(idx + BUF_LEN, iEnd); ++idx2) {
          if (idx2 + offset < sampleLen) {
            input[idx2] += charge*buf2[idx2 - idx];
          }
        }
      }
    } // for each particle
    } // omp parallel
    /*for (int i = 0; i < sizeof(dhfr_particles)/sizeof(dhfr_particles[0]); ++i) {
      double charge = dhfr_particles[i][0];
      double x = dhfr_particles[i][1];
      double y = dhfr_particles[i][2];
      double z = dhfr_particles[i][3];

      for (size_t idx = 0; idx < localLen; ++idx) {
        double x0 = LENGTH*(idx + offset)/globalLen - LENGTH/2;
        double r2 = (x - x0)*(x - x0) + y*y + z*z;
        input[idx] += charge*exp(-r2/SIGMA);
      }
    }*/
  }
  else if (1 == kind) { // e^(i j theta)
#pragma omp parallel for
    for (size_t idx = 0; idx < localLen; idx++) {
      double arg = (idx + offset)*theta;
      input[idx] = COS(arg) + I*SIN(arg);
    }
  }
  else if (0 == kind) { // rectangular function with relative width W
    const long W = globalLen/invRelativeW + 1;
#pragma omp parallel for
    for (size_t idx = 0; idx < localLen; idx++) {
      input[idx] = (2*(idx + offset) < W || 2*(globalLen - idx - offset) < W) ? 1./W : 0;
    }
  }
  else if (3 == kind) { // sinc function
#pragma omp parallel for
    for (size_t idx = 0; idx < localLen; ++idx) {
      if (0 == idx + offset) {
        input[idx] = 1;
      }
      else {
        const long W = globalLen/invRelativeW + 1;
        double A = (double)(idx + offset)*(double)W/(double)globalLen;
        long L = (long)(A + .5);
        double f = ((double)(idx + offset)*(double)W - (double)L*(double)globalLen)/(double)globalLen;
        long odd = L&1;

        input[idx] = (odd ? -SIN(VERIFY_PI*f) : SIN(VERIFY_PI*f))/((2*(idx + offset) <= globalLen) ? W*SIN(VERIFY_PI*(idx + offset)/globalLen) : W*SIN(VERIFY_PI*(globalLen - (idx + offset))/globalLen));
      }
    }
  }
  else {
    assert(0);
  }
}

static cfft_complex_t *output2 = NULL;

cfft_complex_t reference_output(size_t idx, size_t globalLen, int kind, size_t offset) {
  if (2 == kind || 4 == kind || 5 == kind) {
    return output2[idx - offset];
  }
  else if (1 == kind) {
    double arg = theta*globalLen;
    double complex num = 1 - COS(arg) - I*SIN(arg);
    double sin = SIN((theta - 2*VERIFY_PI*idx/globalLen)/2);
    double complex den = 2*sin*sin - I*SIN(theta - 2*VERIFY_PI*idx/globalLen);
    return num/den;
  }
  else if (0 == kind) {
    if (0 == idx) {
      return 1;
    }
    else {
      const long W = globalLen/invRelativeW + 1;
      double A = (double)idx*(double)W/(double)globalLen;
      long L = (long)(A + .5);
      double f = ((double)idx*(double)W - (double)L*(double)globalLen)/(double)globalLen;
      long odd = L&1;

      return (odd ? -SIN(VERIFY_PI*f) : SIN(VERIFY_PI*f))/((2*idx <= globalLen) ? W*SIN(VERIFY_PI*idx/globalLen) : W*SIN(VERIFY_PI*(globalLen - idx)/globalLen));
      //return SIN(VERIFY_PI*idx*(1.0q/invRelativeW + 1.0q/globalLen))/(W*SIN(VERIFY_PI*idx/globalLen));
    }
  }
  else if (3 == kind) {
    const long W = globalLen/invRelativeW + 1;
    return (2*idx < W || 2*(globalLen - idx) < W) ? 1./W : 0;
  }
  else {
    assert(0);
    return 0;
  }
}

double compute_snr(
  cfft_complex_t *output, size_t localLen, size_t offset, size_t globalLen, int kind,
  soi_desc_t *d) {
  if ((2 == kind || 4 == kind || 5 == kind) && output2 == NULL) {
    assert(d);

    DFTI_DESCRIPTOR_DM_HANDLE desc;
    ASSERT_DFTI( DftiCreateDescriptorDM(MPI_COMM_WORLD, &desc, DFTI_TYPE, DFTI_COMPLEX, 1, globalLen) );

    MKL_LONG size;
    ASSERT_DFTI( DftiGetValueDM(desc, CDFT_LOCAL_SIZE, &size) );
    cfft_complex_t *tempBuf =
      (cfft_complex_t *)malloc(sizeof(cfft_complex_t)*size); // FIXME
    if (NULL == tempBuf) {
      fprintf(stderr, "Failed to allocate tempBuf\n");
      exit(1);
    }

    ASSERT_DFTI( DftiSetValueDM(desc, DFTI_PLACEMENT, DFTI_INPLACE) );
    ASSERT_DFTI( DftiCommitDescriptorDM(desc) );

    populate_input(tempBuf, globalLen/d->P, d->PID*globalLen/d->P, globalLen, kind);

    ASSERT_DFTI( DftiComputeForwardDM(desc, tempBuf) );

    ASSERT_DFTI( DftiFreeDescriptorDM(&desc) );

    // repartition data to match with SOI output
    int nSegments =
      d->segmentBoundaries[d->PID + 1] - d->segmentBoundaries[d->PID];
    output2 = (cfft_complex_t *)malloc(sizeof(cfft_complex_t)*nSegments*d->M);

    MPI_Datatype segmentType;
    MPI_Type_contiguous(d->M*2, MPI_TYPE, &segmentType); // *2 for complex
    MPI_Type_commit(&segmentType);

    int sendCnts[d->P], recvCnts[d->P], sendDispls[d->P], recvDispls[d->P];
    for (int p = 0; p < d->P; ++p) {
      int firstSegmentToSend = MAX(d->k*d->PID, d->segmentBoundaries[p]);
      int firstSegmentToRecv = MAX(d->segmentBoundaries[d->PID], d->k*p);

      sendCnts[p] =
        MIN(d->k*(d->PID + 1), d->segmentBoundaries[p + 1]) -
        firstSegmentToSend,
      sendCnts[p] = MAX(sendCnts[p], 0);
      sendDispls[p] = firstSegmentToSend - d->k*d->PID;

      recvCnts[p] =
        MIN(d->k*(p + 1), d->segmentBoundaries[d->PID + 1]) -
        firstSegmentToRecv,
      recvCnts[p] = MAX(recvCnts[p], 0);
      recvDispls[p] = firstSegmentToRecv - d->segmentBoundaries[d->PID];
    }

    CFFT_ASSERT_MPI(MPI_Alltoallv(
      tempBuf, sendCnts, sendDispls, segmentType,
      output2, recvCnts, recvDispls, segmentType,
      MPI_COMM_WORLD));

    free(tempBuf);

    /*DftiCreateDescriptor(&desc, DFTI_TYPE, DFTI_COMPLEX, 1, globalLen);
    DftiCommitDescriptor(desc);

    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    size_t localLen = (globalLen + p - 1)/p;

    for (int i = 0; i < p; ++i) {
      size_t offset = min(localLen*i, globalLen);
      size_t len = min(offset + localLen, globalLen) - offset;

      populate_input(output2 + offset, len, offset, globalLen, kind);
    }

    DftiComputeForward(desc, output2);
    DftiFreeDescriptor(&desc);*/
  }

  double powerErr = 0, powerSig = 0;
#pragma omp parallel for reduction(+:powerErr,powerSig)
  for (size_t idx = 0; idx < localLen; idx++) {
    cfft_complex_t ref = reference_output(idx + offset, globalLen, kind, offset);
    cfft_complex_t diff = output[idx] - ref;
    //printf("(%g %g) (%g %g)\n", creal(ref), cimag(ref), creal(output[idx]), cimag(output[idx]));
    powerErr += creal(diff)*creal(diff) + cimag(diff)*cimag(diff);
    powerSig += creal(ref)*creal(ref) + cimag(ref)*cimag(ref);
  }

  /*int PID;
  MPI_Comm_rank(MPI_COMM_WORLD, &PID); 
  printf("%d err=%g sig=%g snr=%g\n", PID, powerErr, powerSig, 10*log10(powerSig/powerErr));*/

  double globalPowerErr, globalPowerSig;
  MPI_Reduce(&powerErr, &globalPowerErr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&powerSig, &globalPowerSig, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return 10*log10(globalPowerSig/globalPowerErr);
}

double compute_normalized_inf_norm(
  cfft_complex_t *output, size_t localLen, size_t offset, size_t globalLen, int kind) {
  /*int PID;
  MPI_Comm_rank(MPI_COMM_WORLD, &PID); */

  double maxErr = 0, maxSig = 0;
  size_t maxErrIdx = -1;
#pragma omp parallel for reduction(max:maxErr,maxSig)
  for (size_t idx = 0; idx < localLen; idx++) {
    cfft_complex_t ref = reference_output(idx + offset, globalLen, kind, offset);
    cfft_complex_t diff = output[idx] - ref;
    VAL_TYPE diff_mag = creal(diff)*creal(diff) + cimag(diff)*cimag(diff);
    if (diff_mag > maxErr) {
      maxErrIdx = idx;
      maxErr = diff_mag;

      /*if (0 == PID && 3072 == idx) {
        printf("expected:(%g %g), actual:(%g %g), diff=%g\n", creal(ref), cimag(ref), creal(output[idx]), cimag(output[idx]), diff_mag);
      }*/
    }
    maxSig = MAX(creal(ref)*creal(ref) + cimag(ref)*cimag(ref), maxSig);
  }

  //printf("%d %ld %g\n", PID, maxErrIdx, maxErr);

  double globalMaxErr, globalMaxSig;
  MPI_Reduce(&maxErr, &globalMaxErr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&maxSig, &globalMaxSig, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  return sqrt(globalMaxErr/globalMaxSig);
}

}
