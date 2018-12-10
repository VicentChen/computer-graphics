#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call) do { \
  const cudaError_t error = call; \
  if (error != cudaSuccess) { \
    fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
    fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
  } \
}while(0)

#ifdef __linux__
#include <sys/time.h>
inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
#elif _WIN32
#include <Windows.h>
inline unsigned long long seconds() {
  unsigned long long time = GetTickCount64();
  return time;
}
#endif

#endif // _COMMON_H
