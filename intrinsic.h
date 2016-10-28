#ifndef _SOI_FFT_INTRINSIC_H_
#define _SOI_FFT_INTRINSIC_H_

#ifndef PRECISION
#define PRECISION 2
#endif

#if PRECISION == 2
#define SIMD_WIDTH 4
#define SIMDFPTYPE __m256d

#define _MM_LOAD(a) _mm256_load_pd((VAL_TYPE *)(a))
#define _MM_LOADU _mm256_loadu_pd

#define _MM_STORE(a, v) _mm256_store_pd((VAL_TYPE *)(a), v)
#define _MM_STOREU(a, v) _mm256_storeu_pd((VAL_TYPE *)(a), v)
#define _MM_MASKSTORE _mm256_maskstore_pd
#define _MM_STREAM(a, v) _mm256_stream_pd((VAL_TYPE *)(a), v)
#define _MM_PREFETCH1(a) _mm_prefetch((char *)(a), _MM_HINT_T0)

#define _MM_ADD _mm256_add_pd
#define _MM_MUL _mm256_mul_pd
#define _MM_ADDSUB _mm256_addsub_pd
#ifdef __AVX2__
#define _MM_FMADDSUB _mm256_fmaddsub_pd
#else
#define _MM_FMADDSUB(a, b, c) _MM_ADDSUB(_MM_MUL(a, b), c)
#endif

#define _MM_SWAP_REAL_IMAG(a) _mm256_permute_pd(a, 0x05)
#define _MM_MOVELDUP _mm256_movedup_pd
#define _MM_MOVEHDUP(a) _mm256_permute_pd(a, 0xf)

#else
// PRECISION == 1

#define SIMD_WIDTH 8
#define SIMDFPTYPE __m256

#define _MM_LOAD _mm256_load_ps
#define _MM_LOADU _mm256_loadu_ps

#define _MM_STORE _mm256_store_ps
#define _MM_STOREU _mm256_storeu_ps
#define _MM_MASKSTORE _mm256_maskstore_ps
#define _MM_STREAM _mm256_stream_ps

#define _MM_ADD _mm256_add_ps
#define _MM_MUL _mm256_mul_ps
#define _MM_ADDSUB _mm256_addsub_ps

#define _MM_SWAP_REAL_IMAG(a) _mm256_permute_ps(a, 0xb1)
#define _MM_MOVELDUP _mm256_moveldup_ps
#define _MM_MOVEHDUP _mm256_movehdup_ps

#endif // PRECISION == 1

#endif // _SOI_FFT_INTRINSIC_H_
