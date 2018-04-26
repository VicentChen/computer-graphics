#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

int main(int argc, char* argv[]) {
  cl_platform_id *platforms;
  cl_uint num_platforms;
  cl_int i, err, platform_index = -1;

  char* ext_data; size_t ext_size;
  const char icd_ext[] = "cl_khr_icd";

  clGetPlatformIDs(1, NULL, &num_platforms);
  platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  for (i = 0; i < num_platforms; ++i) {
    clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
    ext_data = (char*)malloc(ext_size);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, ext_size, ext_data, NULL);
    printf("Platform %d supports extensions: %s\n", i, ext_data);
    free(ext_data);
  }

  free(platforms);
  return 0;
}
