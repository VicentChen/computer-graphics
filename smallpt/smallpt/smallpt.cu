#define USE_SVPNG

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#ifdef USE_CIMG
#include <CImg.h>
#endif
#ifdef USE_SVPNG
#include <svpng.inc>
#endif

#define M_PI 3.1415926
#define M_1_PI  (1 / M_PI);
#define W 512
#define H 512
#define S 16
#ifdef USE_CIMG
#define COLOR_R (W * H * 0)
#define COLOR_G (W * H * 1)
#define COLOR_B (W * H * 2)
#else
#define COLOR_R 0
#define COLOR_G 1
#define COLOR_B 2
#endif
#define STOP_DEPTH 20

#define CHECK(call) do { \
  const cudaError_t error = call; \
  if (error != cudaSuccess) { \
    fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
    fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
  } \
}while(0)

using namespace std;
#ifdef IMG_SHOW
using namespace cimg_library;
#endif

enum Refl_t { DIFF, SPEC, REFR };

struct Vec {
  double x, y, z;
  __host__ __device__ Vec(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
  __host__ __device__ Vec operator+(const Vec& v) const { return Vec(x + v.x, y + v.y, z + v.z); }
  __host__ __device__ Vec operator-(const Vec& v) const { return Vec(x - v.x, y - v.y, z - v.z); }
  __host__ __device__ Vec operator*(double a)     const { return Vec(x * a, y * a, z * a); }
  __host__ __device__ Vec operator%(const Vec& v) const { return Vec(y * v.z - v.y * z, z * v.x - v.z * x, x * v.y - v.x * y); } // cross product
  __host__ __device__ double dot(const Vec& v) const { return x * v.x + y * v.y + z * v.z; } // dot product
  __host__ __device__ Vec mult(const Vec& v) { return Vec(x * v.x, y * v.y, z * v.z); }
  __host__ __device__ Vec& norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
};

struct Ray {
  Vec o, d; // start position, direction
  __host__ __device__ Ray(Vec o, Vec d) : o(o), d(d) {}
};

struct Sphere {
  double r; // radius
  Vec p, e, c; // position, emission, color
  Refl_t refl; // reflection type

  __host__ __device__ Sphere(double r, Vec p, Vec e, Vec c, Refl_t refl) : r(r), p(p), e(e), c(c), refl(refl) {}

  __host__ __device__ double intersect(const Ray& ray) const {
    //Vec op = p - ray.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
    //double t, eps = 1e-4, b = op.dot(ray.d), det = b * b - op.dot(op) + r * r;
    //if (det < 0) return 0; else det = sqrt(det);
    //return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    Vec op = p - ray.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    double t, eps = 1e-4, b = op.dot(ray.d), det = b * b - op.dot(op) + r * r;
    if (det < 0) return 0; else det = sqrt(det);
    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

static __inline__ __device__ double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
static __inline__ __device__ int toInt(double c) { return floor(pow(clamp(c), 1 / 2.2) * 255 + .5); }

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

static __inline__ __device__ bool intersect(Sphere *d_spheres, const Ray& ray, double &t, int &id) {
  int n = 9;
  double d;
  double inf = t = 1e20; // infinity
  // find the closest interset point
  for (int i = 0; i < n; i++) {
    d = d_spheres[i].intersect(ray);
    if (d && d < t) {
      t = d;
      id = i;
    }
  }
  return t < inf;
}

static __inline__ __device__ Ray diffuse_ray(Ray& in, Vec& x, Vec& N_up, curandState *state) {
  double angle = 2 * M_PI * curand_uniform_double(state), r2 = curand_uniform_double(state), r2s = sqrt(r2);
  // axis: u = x, v = y, w = z
  Vec w = N_up, u = (abs(w.x) > 0.1 ? Vec(0, 1) : Vec(1) % w), v = w % u;
  Vec reflect_d = (u * cos(angle) * r2s + v * sin(angle) * r2s + w * sqrt(1 - r2)).norm();
  return Ray(x, reflect_d);
}

static __inline__ __device__ Ray refract_ray(Ray& ray, Vec& x, Vec& N, Vec& N_up, double& rate, curandState *state) {
  Ray reflect_ray(x, ray.d - N * 2 * N.dot(ray.d));
  bool into = N_up.dot(N) > 0;
  double na = 1, ng = 1.5; // IOR(index of refraction) for air and glass
  double ior = into ? na / ng : ng / na; // IOR
  double cos_i = ray.d.dot(N_up); // incident angle
  double cos_r2 = 1 - ior * ior * (1 - cos_i * cos_i); // refraction angle

  if (cos_r2 < 0) { // total reflection
    ray = reflect_ray;
  }
  else { // refraction
    Vec refract_d = (ray.d * ior - N * ((into ? 1 : -1) * (cos_i * ior + sqrt(cos_r2)))).norm();
    // Schlick's approximation
    double R0 = (ng - na) * (ng - na) / (ng + na) / (ng + na);
    double c = 1 - (into ? -cos_i : refract_d.dot(N));
    double R = R0 + (1 - R0) * c * c * c * c * c;
    double T = 1 - R;
    double P = .25 + .5 * R;
    double RP = R / P, TP = T / (1 - P);

    if (curand_uniform_double(state) < P) {
      rate = RP;
      return reflect_ray;
    }
    else {
      rate = TP;
      return Ray(x, refract_d);
    }
  }
}

static __inline__ __device__ Vec radiance(Sphere *d_spheres, const Ray &r, int depth, curandState *state) {

  double t;
  int id;

  Ray ray = r;

  Vec cl(0, 0, 0); // accumulated L
  Vec cf(1, 1, 1); // accumulated F

  for (; depth < STOP_DEPTH;) { // cannot be infinite loop
    if (!intersect(d_spheres, ray, t, id)) return cl;// if not intersect return dark
    const Sphere& obj = d_spheres[id];

    // obj properties
    Vec x = ray.o + ray.d * t;
    Vec N = (x - obj.p).norm(); // normal vector of intersect point on sphere
    Vec N_up = ray.d.dot(N) < 0 ? N : N * -1;
    Vec f = obj.c;
    double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;
    double rate;

    // accumulate color
    cl = cl + cf.mult(obj.e);
    if (++depth > 5 || !p) { // Russian Roulette
      if (curand_uniform_double(state) < p) f = f * (1 / p);
      else return cl;
    }
    cf = cf.mult(f);

    switch (obj.refl) {
      case SPEC:
        {
          ray = Ray(x, ray.d - N * 2 * N.dot(ray.d));
          break;
        }
      case REFR:
        {
          ray = refract_ray(ray, x, N, N_up, rate, state);
          cf = cf * rate;
          break;
        }
      default:
        {
          ray = diffuse_ray(ray, x, N_up, state);
          break;
        }
    }
  }
  return cl;
}

__global__ void render(Sphere *d_spheres, Vec *color, Ray *cam, Vec *cx, Vec *cy) {
  int u = blockIdx.x;
  int v = blockIdx.y;
  int w = u;
  int h = H - v - 1;
  int block_id = v * gridDim.x + u;
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

  curandState local_state;
  curand_init(block_id, thread_id, 0, &local_state);
  Vec r;
  for (int s = 0; s < S; s++) {
    double a = M_PI * 2 * thread_id / (S * S);
    double rx = .5 * cos(a) + .5;//2 * curand_uniform_double(&local_state); rx = rx < 1 ? sqrt(rx) - 1 : 1 - sqrt(2-rx); rx += .5;
    double ry = .5 * sin(a) + .5;//2 * curand_uniform_double(&local_state); ry = ry < 1 ? sqrt(ry) - 1 : 1 - sqrt(2-ry); ry += .5;
    Vec d = *cx * ((w + rx) / W - .5) + *cy * ((h + ry) / H - .5) + cam->d;
    r = r + radiance(d_spheres, Ray(cam->o + d * 138, d.norm()), 0, &local_state);
  }
  r = r * (1. / (S));
  atomicAdd(&(color[v * W + u].x), r.x);
  atomicAdd(&(color[v * W + u].y), r.y);
  atomicAdd(&(color[v * W + u].z), r.z);
}

__global__ void compute_img(Vec *color, unsigned char *I) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  Vec &r = color[idx];
#ifndef USE_CIMG
  idx *= 3;
#endif
  I[idx + COLOR_R] = toInt(clamp(r.x / (S * S)));
  I[idx + COLOR_G] = toInt(clamp(r.y / (S * S)));
  I[idx + COLOR_B] = toInt(clamp(r.z / (S * S)));
}

int main(int argc, char* argv[]) {
  unsigned char *buffer = (unsigned char*)malloc(W * H * 3);
  Vec *double_color = (Vec*)malloc(sizeof(Vec) * W * H);
  unsigned char *d_buffer;
  Vec *d_double_color;
  curandState *d_state;
  Ray *d_cam;
  Vec *d_cx, *d_cy;
  Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());
  Vec cx = Vec(W * 0.5135 / H), cy = (cx % cam.d).norm() * .5135;
  Sphere *d_spheres;

  Sphere spheres[] = {//Scene: radius, position, emission, color, material 
  Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
  Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
  Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back
  Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF),//Frnt
  Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm
  Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top 
  Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr
  Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas
  Sphere(600, Vec(50,681.6 - .27,81.6),Vec(12,12,12),  Vec(), DIFF) //Lite
  };

  CHECK(cudaSetDevice(0));
  CHECK(cudaMalloc((void**)&d_buffer, W * H * 3));
  CHECK(cudaMalloc((void**)&d_cam, sizeof(Ray)));
  CHECK(cudaMalloc((void**)&d_cx, sizeof(Vec)));
  CHECK(cudaMalloc((void**)&d_cy, sizeof(Vec)));
  CHECK(cudaMalloc((void**)&d_spheres, sizeof(Sphere) * 9));
  CHECK(cudaMalloc((void**)&d_double_color, sizeof(Vec) * W * H));
  CHECK(cudaMalloc((void**)&d_state, sizeof(curandState) * W * H));
  CHECK(cudaMemset(d_double_color, 0, sizeof(Vec) * W * H));
  CHECK(cudaMemcpy(d_cam, &cam, sizeof(Ray), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_cx, &cx, sizeof(Vec), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_cy, &cy, sizeof(Vec), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_spheres, spheres, sizeof(Sphere) * 9, cudaMemcpyHostToDevice));

  dim3 grids(H, W);
  dim3 blocks(S, S);

  render<<<grids, blocks>>>(d_spheres, d_double_color, d_cam, d_cx, d_cy);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  compute_img<<<H, W>>>(d_double_color, d_buffer);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(buffer, d_buffer, W * H * 3, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(double_color, d_double_color, sizeof(Vec) * W * H, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_cam));
  CHECK(cudaFree(d_cx));
  CHECK(cudaFree(d_cy));
  CHECK(cudaFree(d_buffer));
  CHECK(cudaFree(d_spheres));
  CHECK(cudaFree(d_double_color));
  CHECK(cudaFree(d_state));
  CHECK(cudaDeviceReset());

  char *filename = "output-cuda.png";

#ifdef USE_CIMG
  CImg<unsigned char> img(buffer, W, H, 1, 3, true);
  img.display();
  img.save_png(filename);
#endif

#ifdef USE_SVPNG
  FILE* file = fopen(filename, "wb");
  svpng(file, W, H, buffer, 0);
  fclose(file);
#endif

  free(buffer);
  free(double_color);
}
