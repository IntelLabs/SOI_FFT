#pragma once

int max_exponent(const double *x, int len);

/**
 * @ret the length of compressed data w.r.t. number of integers
 */
int compress(int *out, const double *in, int len, int e_max, int e_max_i, int print);

void decompress(double *out, const int *in, int len, int e_max, int e_max_i, const double *refIn);
