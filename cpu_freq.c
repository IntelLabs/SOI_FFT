#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#include "soi.h"

static const double DEFAULT_CPU_FREQ = 2.2e9;

double get_cpu_freq()
{
  static double freq = DBL_MAX;
  if (DBL_MAX == freq) {
    volatile double a = rand()%1024, b = rand()%1024;
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    unsigned long long t1 = __rdtsc();
    for (size_t i = 0; i < 1024L*1024; i++) {
      a += a*b + b/a;
    }
    unsigned long long dt = __rdtsc() - t1;
    gettimeofday(&tv2, NULL);
    freq = dt/((tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec)/1.e6);
  }

  return freq;
}
/*double get_cpu_freq()
{
  FILE *fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "rt");
  if (fp == NULL) {
    //printf("Can't open /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq\n"); fflush(0);
    return DEFAULT_CPU_FREQ;
  }
  char buf[1024];
  fgets(buf, sizeof(buf), fp);
  fclose(fp);
  return atof(buf)*1000;
}*/
