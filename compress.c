#include <math.h>
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>

#include "soi.h"
#include "compress.h"

int max_exponent(const double *x, int len)
{
  int global_max_ref = INT_MIN;
  int maxs[256];

/*#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int iPerThread = (len + nthreads - 1)/nthreads;
    int iBegin = iPerThread*tid;
    int iEnd = MIN(iBegin + iPerThread, len);

    int max = INT_MIN;

    for (int i = iBegin; i < iEnd; ++i) {
      unsigned long long ull = *(unsigned long long *)(x + i);
      int e = ((ull >> 52ULL) & ((1 << 11) - 1));
      max = MAX(e, max);
    }

    maxs[tid] = max;

#pragma omp barrier
#pragma omp single
    {
      for (int i = 0; i < nthreads; ++i) {
        global_max_ref = MAX(global_max_ref, maxs[i]);
      }
    }
  }

  int global_max = global_max_ref;*/

  int global_max = INT_MIN;
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int iPerThread = (len/8 + nthreads - 1)/nthreads;
    int iBegin = iPerThread*8*tid;
    int iEnd = MIN(iBegin + iPerThread*8, len);

    __m256i max_v = _mm256_set1_epi32(INT_MIN);

    for (int i = iBegin; i < iEnd; i += 8) {
      __m256 l1 = _mm256_load_ps((float *)(x + i));
      __m256 l2 = _mm256_load_ps((float *)(x + i + 4));

      __m256i l_v = _mm256_castps_si256(_mm256_shuffle_ps(l1, l2, _MM_SHUFFLE(3, 1, 3, 1)));
      __m256i e_v = _mm256_srli_epi32(_mm256_slli_epi32(l_v, 1), 21);
        // take the first (32-21) = 11 bits after the sign bit.

      max_v = _mm256_max_epi32(e_v, max_v);
    }

    __declspec(align(32)) int tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, max_v);
      // by some reason, icc doesn't align tmp properly
      // so we use unaligned store
    int max = INT_MIN;
    for (int i = 0; i < 8; ++i) {
      max = MAX(tmp[i], max);
    }

    maxs[tid] = max;

#pragma omp barrier
#pragma omp single
    {
      for (int i = 0; i < nthreads; ++i) {
        global_max = MAX(global_max, maxs[i]);
      }

      for (int i = len/8*8; i < len; ++i) {
        unsigned long long ull = *(unsigned long long *)(x + i);
        int e = ((ull >> 52ULL) & ((1 << 11) - 1));
        global_max = MAX(e, global_max);
      }
    }
  }

  //printf("(%d, %d)\n", global_max_ref, global_max);

#if 0
//#pragma omp parallel for reduction(max:max)
  for (int i = 0; i < len; i++) {
    unsigned long long ull = *(unsigned long long *)(x + i);
    int e = ((ull >> 52ULL) & ((1 << 11) - 1)) - 1023;
    printf("%d\n", e);
    max = e > max ? e : max;
  }

  __m256i max_v = _mm256_set1_epi32(INT_MIN);

//#pragma omp parallel for
  for (int i = 0; i < len; i += 8) {
    __m256 l1 = _mm256_load_ps((float *)(x + i));
    __m256 l2 = _mm256_load_ps((float *)(x + i + 4));

    __m256i l_v = _mm256_castps_si256(_mm256_shuffle_ps(l1, l2, _MM_SHUFFLE(3, 1, 3, 1)));
    __m256i e_v = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_slli_epi32(l_v, 1), 21), _mm256_set1_epi32(1023));

    max_v = _mm256_max_epi32(e_v, max_v);

    __declspec(align(64)) int tmp[8];
    _mm256_store_si256((__m256i *)tmp, e_v);
    for (int j = 0; j < 8; ++j) {
      unsigned long long ull = *(unsigned long long *)(x + i + j);
      int e = ((ull >> 52ULL) & ((1 << 11) - 1)) - 1023;

      printf("(%d, %d) ", tmp[j/*j%2*4 + j/2*/], e);
    }
    printf("\n");
  }
#endif

  return global_max - 1023;
}

static inline int logicalRightShift(int i, int shft)
{
  if (shft > 0) {
    return (i >> shft) & ((1 << (32 - shft)) - 1);
  }
  else {
    return i;
  }
}

//#define DEBUG_VLC

static const int NBITS = 52;

static inline void append(
  int i, int *curWord, int *curWordOccupancy, int *out, int *outIdx, int nbits)
{
  i &= (1UL << nbits) - 1; // take nbits of i
  *curWordOccupancy += nbits; // update occupancy
  if (*curWordOccupancy >= 32) { // if overflows the current word
    nbits = *curWordOccupancy - 32;
    *curWord |= logicalRightShift(i, nbits);
    out[*outIdx] = *curWord;
#ifdef DEBUG_VLC
    printf("write %0x\n", *curWord);
#endif
    *curWord = 0;
    i &= (1U << nbits) - 1;

    ++(*outIdx);
    *curWordOccupancy = nbits;
  }
  *curWord |= i << (32 - *curWordOccupancy);
}

#define VLEN (16)

static inline void append8(
  __m256i *i, __m256i *curWord, int *curWordOccupancy, int *out, int *outIdx, int nbits)
{
  i[0] = _mm256_and_si256(i[0], _mm256_set1_epi32((1UL << nbits) - 1));
  i[1] = _mm256_and_si256(i[1], _mm256_set1_epi32((1UL << nbits) - 1));

  *curWordOccupancy += nbits;

  if (*curWordOccupancy >= 32) {
    nbits = *curWordOccupancy - 32;

    _mm256_store_si256(
      (__m256i *)(out + *outIdx),
      _mm256_or_si256(curWord[0], _mm256_srli_epi32(i[0], nbits)));
    _mm256_store_si256(
      (__m256i *)(out + *outIdx + VLEN/2),
      _mm256_or_si256(curWord[1], _mm256_srli_epi32(i[1], nbits)));
    (*outIdx) += VLEN;

    i[0] = _mm256_and_si256(i[0], _mm256_set1_epi32((1U << nbits) - 1));
    i[1] = _mm256_and_si256(i[1], _mm256_set1_epi32((1U << nbits) - 1));

    *curWordOccupancy = nbits;

    curWord[0] = _mm256_slli_epi32(i[0], 32 - *curWordOccupancy);
    curWord[1] = _mm256_slli_epi32(i[1], 32 - *curWordOccupancy);
  }
  else {
    curWord[0] = _mm256_or_si256(
      curWord[0], _mm256_slli_epi32(i[0], 32 - *curWordOccupancy));
    curWord[1] = _mm256_or_si256(
      curWord[1], _mm256_slli_epi32(i[1], 32 - *curWordOccupancy));
  }
}

int compress(int *out, const double *in, int len, int e_max, int e_max_i, int print)
{
  const int i1Len = (NBITS + 3 - 31) - (e_max - e_max_i);
  const int i2Len = MIN(32, NBITS + 3 - (e_max - e_max_i));

  if (i2Len <= 0) {
    return 0;
  }

  double e1 = pow(2, (NBITS - 31) - e_max);
  double e2 = pow(2, NBITS - e_max);
  double e3 = pow(2, 31);

  int curWord[VLEN] = { 0 };
  int curWordOccupancy = 0;
  int outIdx = 0;

  __m256i curWordV[2];
  curWordV[0] = _mm256_setzero_si256();
  curWordV[1] = _mm256_setzero_si256();

  int i = 0;
  for ( ; i < len/VLEN*VLEN; i += VLEN) {
#define VECTOR
#ifdef VECTOR
    __m256d x1 = _mm256_load_pd(in + i);
    __m256d x2 = _mm256_load_pd(in + i + VLEN/4);
    __m256d x3 = _mm256_load_pd(in + i + VLEN/4*2);
    __m256d x4 = _mm256_load_pd(in + i + VLEN/4*3);

    __m256d d11 = _mm256_round_pd(
      _mm256_mul_pd(x1, _mm256_set1_pd(e1)),
      _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256d d12 = _mm256_fmsub_pd(
      x1, _mm256_set1_pd(e2),
      _mm256_mul_pd(d11, _mm256_set1_pd(e3)));

    __m256d d21 = _mm256_round_pd(
      _mm256_mul_pd(x2, _mm256_set1_pd(e1)),
      _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256d d22 = _mm256_fmsub_pd(
      x2, _mm256_set1_pd(e2),
      _mm256_mul_pd(d21, _mm256_set1_pd(e3)));

    __m256d d31 = _mm256_round_pd(
      _mm256_mul_pd(x3, _mm256_set1_pd(e1)),
      _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256d d32 = _mm256_fmsub_pd(
      x3, _mm256_set1_pd(e2),
      _mm256_mul_pd(d31, _mm256_set1_pd(e3)));

    __m256d d41 = _mm256_round_pd(
      _mm256_mul_pd(x4, _mm256_set1_pd(e1)),
      _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256d d42 = _mm256_fmsub_pd(
      x4, _mm256_set1_pd(e2),
      _mm256_mul_pd(d41, _mm256_set1_pd(e3)));

    __m256i i1[2], i2[2];
    i1[0] = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm256_cvtpd_epi32(d11)),
      _mm256_cvtpd_epi32(d21),
      1);
    i2[0] = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm256_cvtpd_epi32(d12)),
      _mm256_cvtpd_epi32(d22),
      1);

    i1[1] = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm256_cvtpd_epi32(d31)),
      _mm256_cvtpd_epi32(d41),
      1);
    i2[1] = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm256_cvtpd_epi32(d32)),
      _mm256_cvtpd_epi32(d42),
      1);
#else
    __m256i i1[2], i2[2];

    __declspec(aligned(64)) int i1_[VLEN], i2_[VLEN];

    double e1_decompress = pow(2, e_max - (NBITS - 31));
    double e2_decompress = pow(2, e_max - NBITS);

    for (int j = 0; j < VLEN; ++j) {
      double x = in[i + j];
      double d1 = round(x*e1);
      double d2 = round(x*e2 - d1*e3);
      unsigned long long xx = *((unsigned long long *)&x);
      i1_[j] = (int)d1;
      i2_[j] = (int)d2;

      if (print) {
        //if (260 == i + j) {
          printf("e_max = %d, e1 = %g, e2 = %g, e3 = %g, x*e1=%g, x*e2 - d1*e3 = %g, x = %g, d1 = %g, d2 = %g, i1 = 0x%x, i2 = 0x%x\n", e_max, e1, e2, e3, x*e1, x*e2 - d1*e3, x, d1, d2, i1_[j], i2_[j]);
        //}
        printf(
          "%d: %g -> %g\n",
          i + j,
          x,
          (double)i1_[j]*e1_decompress + (double)i2_[j]*e2_decompress);
      }
    }

    i1[0] = _mm256_load_si256((__m256i *)i1_);
    i1[1] = _mm256_load_si256((__m256i *)(i1_ + 8));

    i2[0] = _mm256_load_si256((__m256i *)i2_);
    i2[1] = _mm256_load_si256((__m256i *)(i2_ + 8));
#endif

    // take first up to (32 - curWordOccupancy) bits of last (23 - (e_max - e^i_max) bits of i1
    if (i1Len > 0) {
      append8(i1, curWordV, &curWordOccupancy, out, &outIdx, i1Len);
    }

    append8(i2, curWordV, &curWordOccupancy, out, &outIdx, i2Len);
  }

  if (curWordOccupancy) {
    _mm256_store_si256((__m256i *)(out + outIdx), curWordV[0]);
    _mm256_store_si256((__m256i *)(out + outIdx + VLEN/2), curWordV[1]);
    outIdx += VLEN;
  }

  curWord[0] = 0;
  curWordOccupancy = 0;

  for ( ; i < len; ++i) {
    double x = in[i];
    double d1 = round(x*e1);
      // left shift by (52 - 31) - e_max
    double d2 = round(x*e2 - d1*e3);
      // when d1 is not zero : -2^30 < d2 <= d^30 -> requires 32 bits (TODO reduce it to 31 bits)
      // when d1 is zero: requires 52 + 2 - (e_max - e_max_i) bits
    unsigned long long xx = *((unsigned long long *)&x);
    int i1 = (int)d1;
    int i2 = (int)d2;
    /*printf(
      "%d %g (%A) %g (%A) %g (%A) %llx %llx %x %x %x %x\n",
      i, x, x, d1, d1, d2, d2,
      xx, xx&((1LL << 52) - 1),
      (int)(xx&((1U << 31) - 1)),
      (int)((xx >> 31)&((1 << 21) - 1)),
      i1, i2);*/

#ifdef DEBUG_VLC
    printf("%d compress: i1 = %x, i2 = %x\n", i, i1, i2);
#endif

    // take first up to (32 - curWordOccupancy) bits of last (23 - (e_max - e^i_max) bits of i1
    if (i1Len > 0) {
      append(i1, curWord, &curWordOccupancy, out, &outIdx, i1Len);
    }

    append(i2, curWord, &curWordOccupancy, out, &outIdx, i2Len);
  }

  if (curWordOccupancy) {
    out[outIdx] = curWord[0];
    ++outIdx;
#ifdef DEBUG_VLC
    printf("write %0x\n", curWord[0]);
#endif
  }

  return outIdx;
}

int extract(
  int *curWord, int *curWordOccupancy, const int *in, int *inIdx, int nbits)
{
  int i = (*curWord << (32 - *curWordOccupancy)) >> (32 - nbits); // extract nbits
  if (nbits <= *curWordOccupancy) { // if we get all the bits we need
    *curWordOccupancy -= nbits;
    if (*curWordOccupancy == 0) {
      *curWord = in[*inIdx];
      ++(*inIdx);
      *curWordOccupancy = 32;
    }
  }
  else { // if we also need to extract from the next word
    *curWord = in[*inIdx];
    i |= logicalRightShift(*curWord, 32 - (nbits - *curWordOccupancy));
    ++(*inIdx);
    *curWordOccupancy = 32 - (nbits - *curWordOccupancy);
  }
  return i;
}

void extract8(
  __m256i *i, __m256i *curWord, int *curWordOccupancy, const int *in, int *inIdx, int nbits) 
{
  i[0] = _mm256_srai_epi32(
    _mm256_slli_epi32(curWord[0], 32 - *curWordOccupancy), 32 - nbits);
  i[1] = _mm256_srai_epi32(
    _mm256_slli_epi32(curWord[1], 32 - *curWordOccupancy), 32 - nbits);

  if (nbits <= *curWordOccupancy) {
    *curWordOccupancy -= nbits;

    if (*curWordOccupancy == 0) {
      curWord[0] = _mm256_load_si256((__m256i *)(in + *inIdx));
      curWord[1] = _mm256_load_si256((__m256i *)(in + *inIdx + VLEN/2));

      (*inIdx) += VLEN;
      *curWordOccupancy = 32;
    }
  }
  else {
    curWord[0] = _mm256_load_si256((__m256i *)(in + *inIdx));
    curWord[1] = _mm256_load_si256((__m256i *)(in + *inIdx + VLEN/2));

    i[0] = _mm256_or_si256(
      i[0], _mm256_srli_epi32(curWord[0], 32 - (nbits - *curWordOccupancy)));
    i[1] = _mm256_or_si256(
      i[1], _mm256_srli_epi32(curWord[1], 32 - (nbits - *curWordOccupancy)));

    (*inIdx) += VLEN;
    *curWordOccupancy = 32 - (nbits - *curWordOccupancy);
  }
}

void decompress(double *out, const int *in, int len, int e_max, int e_max_i, const double *refIn)
{
  double e1 = pow(2, e_max - (NBITS - 31));
  double e2 = pow(2, e_max - NBITS);

  const int i1Len = (NBITS + 3 - 31) - (e_max - e_max_i);
  const int i2Len = MIN(32, NBITS + 3 - (e_max - e_max_i));

  int curWord[VLEN];

  __m256i curWordV[2];
  curWordV[0] = _mm256_load_si256((__m256i *)in);
  curWordV[1] = _mm256_load_si256((__m256i *)(in + VLEN/2));

  int curWordOccupancy = 32;
  int inIdx = VLEN;

//#define COMPUTE_SNR
#ifdef COMPUTE_SNR
  double powerErr = 0, powerSig = 0;
#endif

  int i = 0;
  for ( ; i < len/VLEN*VLEN; i += VLEN) {
    __m256i i1[2], i2[2];
    i1[0] = _mm256_setzero_si256();
    i1[1] = _mm256_setzero_si256();

    if (i1Len > 0) {
      extract8(i1, curWordV, &curWordOccupancy, in, &inIdx, i1Len);
    }

    extract8(i2, curWordV, &curWordOccupancy, in, &inIdx, i2Len);

    __m256d d11 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i1[0], 0));
    __m256d d12 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i1[0], 1));
    __m256d d13 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i1[1], 0));
    __m256d d14 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i1[1], 1));

    __m256d d21 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i2[0], 0));
    __m256d d22 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i2[0], 1));
    __m256d d23 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i2[1], 0));
    __m256d d24 = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i2[1], 1));

    __m256d x1 = _mm256_fmadd_pd(
      d11, _mm256_set1_pd(e1),
      _mm256_mul_pd(d21, _mm256_set1_pd(e2)));
    __m256d x2 = _mm256_fmadd_pd(
      d12, _mm256_set1_pd(e1),
      _mm256_mul_pd(d22, _mm256_set1_pd(e2)));
    __m256d x3 = _mm256_fmadd_pd(
      d13, _mm256_set1_pd(e1),
      _mm256_mul_pd(d23, _mm256_set1_pd(e2)));
    __m256d x4 = _mm256_fmadd_pd(
      d14, _mm256_set1_pd(e1),
      _mm256_mul_pd(d24, _mm256_set1_pd(e2)));

    _mm256_stream_pd(out + i, x1);
    _mm256_stream_pd(out + i + VLEN/4, x2);
    _mm256_stream_pd(out + i + VLEN/4*2, x3);
    _mm256_stream_pd(out + i + VLEN/4*3, x4);
  }

  if (curWordOccupancy < 32) {
    inIdx += VLEN;
  }
  else {
    inIdx -= VLEN - 1;
  }

  curWord[0] = in[inIdx];
  curWordOccupancy = 32;

  for ( ; i < len; ++i) {
    int i1 = 0;

    int i1Len = (NBITS + 3 - 31) - (e_max - e_max_i);
    if (i1Len > 0) {
      i1 = extract(curWord, &curWordOccupancy, in, &inIdx, i1Len);
    }

    int i2Len = MIN(32, NBITS + 3 - (e_max - e_max_i));
    int i2 = extract(curWord, &curWordOccupancy, in, &inIdx, i2Len);

#ifdef DEBUG_VLC
    printf("%d decompress: i1 = %x, i2 = %x\n", i, i1, i2);

    /*double refD1 = floor(refIn[i]*pow(2, (NBITS - 31) - e_max));
    double refD2 = floor(refIn[i]*pow(2, NBITS - e_max) - refD1*pow(2, 31));
    int refI1 = (int)refD1;
    int refI2 = (int)refD2;
    if (refI1 != i1) {
      printf("%d %x expected %x actual\n", i, refI1, i1);
      assert(0);
    }
    if (refI2 != i2) {
      printf("%d %x expected %x actual\n", i, refI2, i2);
      assert(0);
    }*/
#endif

    double d1 = (double)i1;
    double d2 = (double)i2;
    double x = d1*e1 + d2*e2;
    out[i] = x;
#ifdef COMPUTE_SNR
    powerSig += refIn[i]*refIn[i];
    powerErr += (refIn[i] - x)*(refIn[i] - x);
#endif
    //printf("%d %g %g %g %g\n", i, x, d1, d2, log2(fabs(refIn[i] - x)));
  }
  //printf("inIdx = %d\n", inIdx);
#ifdef COMPUTE_SNR
  printf("snr from compression = %f\n", 10*log10(powerSig/powerErr));
  printf("RMSS = %f\n", sqrt(powerSig/len));
#endif
}

/*int main()
{
  const int SEGMENT_LEN = 4;

  double segments[2][4] = {
    { 0X1.0123456789ABCp8, 0X1.DEF0123456789p8, -0X1.0123456789ABCp8, -0X1.DEF0123456789p8, },
    { 0X1.0123456789ABCp0, 0X1.DEF0123456789p-1, -0X1.0123456789ABCp0, -0X1.DEF0123456789p0, },
  };

  const int NUM_SEGMENTS = sizeof(segments)/sizeof(segments[0]);

  int e_maxs[NUM_SEGMENTS];
  
  int global_e_max = INT_MIN;
  for (int s = 0; s < NUM_SEGMENTS; ++s) {
    e_maxs[s] = max_exponent(segments[s], SEGMENT_LEN);
    global_e_max = e_maxs[s] > global_e_max ? e_maxs[s] : global_e_max;
  }

  int compressed[NUM_SEGMENTS][SEGMENT_LEN*2];
  double decompressed[NUM_SEGMENTS][SEGMENT_LEN];

  for (int s = 0; s < NUM_SEGMENTS; ++s) {
    //printf("segment %d\n", s);
    compress(compressed[s], segments[s], SEGMENT_LEN, global_e_max, e_maxs[s]);
    decompress(decompressed[s], compressed[s], SEGMENT_LEN, global_e_max, e_maxs[s]);
  }

  double expected[2][4] = {
    { 0X1.0123456789ABCp8, 0X1.DEF0123456789p8, -0X1.0123456789ABCp8, -0X1.DEF0123456789p8, },
    { 0X1.0123456789Ap0, 0X1.DEF01234566p-1, -0X1.0123456789Bp0, -0X1.DEF01234568p0, },
  };

  for (int s = 0; s < NUM_SEGMENTS; ++s) {
    for (int i = 0; i < SEGMENT_LEN; ++i) {
      printf("%A ", decompressed[s][i]);
    }
    printf("\n");
  }

  for (int s = 0; s < NUM_SEGMENTS; ++s) {
    for (int i = 0; i < SEGMENT_LEN; ++i) {
      if (decompressed[s][i] != expected[s][i]) {
        printf(
          "expected %A actual %A at segment %d element %d\n", 
          expected[s][i], decompressed[s][i], s, i);
      }
    }
  }

  return 0;
}*/
