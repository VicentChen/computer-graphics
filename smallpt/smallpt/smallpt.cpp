#include <iostream>
#include <cstdlib>
#include <cmath>
#include <CImg.h>

using namespace std;
using namespace cimg_library;

double M_PI = 3.1415926535;
double M_1_PI = 1 / M_PI;
inline double erand48(unsigned short *Xi) { return (double)rand() / (double)RAND_MAX; }

struct Vec {
  double x, y, z;
  Vec(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
  Vec operator+(const Vec& v) const { return Vec(x + v.x, y + v.y, z + v.z); }
  Vec operator-(const Vec& v) const { return Vec(x - v.x, y - v.y, z - v.z); }
  Vec operator*(double a)     const { return Vec(x * a, y * a, z * a); }
  Vec operator%(const Vec& v) const { return Vec(y * v.z - v.y * z, z * v.x - v.z * x, x * v.y - v.x * y); } // cross product
  double dot(const Vec& v) const { return x * v.x + y * v.y + z * v.z; } // dot product
  Vec mult(const Vec& v) { return Vec(x * v.x, y * v.y, z * v.z); }
  Vec& norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
};

struct Ray {
  Vec o, d; // start position, direction
  Ray(Vec o, Vec d) : o(o), d(d) {}
};

// reflection type
enum Refl_t { DIFF, SPEC, REFR };

struct Sphere {
  double r; // radius
  Vec p, e, c; // position, emission, color
  Refl_t refl; // reflection type

  Sphere(double r, Vec p, Vec e, Vec c, Refl_t refl) : r(r), p(p), e(e), c(c), refl(refl) {}

  double intersect(const Ray& ray) const {
    Vec op = p - ray.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
    double t, eps = 1e-4, b = op.dot(ray.d), det = b * b - op.dot(op) + r * r;
    if (det < 0) return 0; else det = sqrt(det);
    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

// scene setup
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

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline int toInt(double c) { return floor(pow(clamp(c), 1 / 2.2) * 255 + .5); }
inline bool intersect(const Ray& ray, double &t, int &id) {
  int n = sizeof(spheres) / sizeof(Sphere);
  double d;
  double inf = t = 1e20; // infinity
  // find the closest interset point
  for (int i = 0; i < n; i++) {
    d = spheres[i].intersect(ray);
    if (d && d < t) {
      t = d;
      id = i;
    }
  }
  return t < inf;
}

Vec radiance(const Ray &r, int depth, unsigned short *Xi) {
  double t;
  int id;

  Ray ray = r;

  Vec cl(0, 0, 0); // accumulated L
  Vec cf(1, 1, 1); // accumulated F

  while (true) {
    if (!intersect(ray, t, id)) return Vec();// if not intersect return dark
    const Sphere& obj = spheres[id];

    // obj properties
    Vec x = ray.o + ray.d * t;
    Vec N = (x - obj.p).norm(); // normal vector of intersect point on sphere
    Vec N_up = ray.d.dot(N) < 0 ? N : N * -1;
    Vec f = obj.c;
    double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;

    // accumulate color
    cl = cl + cf.mult(obj.e);
    cf = cf.mult(f);

    if (++depth > 5 || !p) { // Russian Roulette
      if (erand48(Xi) < p) f = f * (1 / p);
      else return cl;
    }

    if (obj.refl == DIFF) {
      // random reflect direction
      double angle = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
      // axis: u = x, v = y, w = z 
      Vec w = N_up, u = (abs(w.x) > 0.1 ? Vec(0, 1) : Vec(1) % w), v = w % u;
      Vec reflect_d = (u * cos(angle) * r2s + v * sin(angle) * r2s + w * sqrt(1 - r2)).norm();
      ray = Ray(x, reflect_d);
    }
    else if (obj.refl == SPEC) {
      ray = Ray(x, ray.d - N * 2 * N.dot(ray.d));
    }
    else {
      Ray reflect_ray(x, ray.d - N * 2 * N.dot(ray.d));
      //bool into = ray.d.dot(N) < 0;
      bool into = N_up.dot(N) > 0;
      double na = 1, ng = 1.5; // IOR(index of refraction) for air and glass
      double ior = into ? na / ng : ng / na; // IOR
      double cos_i = ray.d.dot(N_up); // incident angle
      double cos_r2 = 1 - ior * ior * (1 - cos_i * cos_i); // refraction angle
      if (cos_r2 < 0) {
        // total reflection
        ray = reflect_ray;
      }
      else {
        // refraction
        Vec refract_d = (ray.d * ior - N * ((into ? 1 : -1) * (cos_i * ior + sqrt(cos_r2)))).norm();
        // Schlick's approximation
        double R0 = (ng - na) * (ng - na) / (ng + na) / (ng + na);
        double c = 1 - (into ? -cos_i : refract_d.dot(N));
        double R = R0 + (1 - R0) * c * c * c * c * c;
        double T = 1 - R;
        double P = .25 + .5 * R;
        double RP = R / P, TP = T / (1 - P);
        
        if(erand48(Xi) < P) {
          cf = cf * RP;
          ray = reflect_ray;
        }
        else {
          cf = cf * TP;
          ray = Ray(x, refract_d);
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  int w = 512, h = 512; // image width, image height
  int samps = argc == 2 ? atoi(argv[1]) / 4 : 16;
  Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());
  Vec cx = Vec(w * 0.5135 / h), cy = (cx % cam.d).norm() * .5135;
  Vec r;
  Vec *c = new Vec[w * h];

#pragma omp parallel for schedule(dynamic, 1) private(r) // OpenMP

  for (int y = 0; y < h; y++) {
    cout << "Rendering line " << y << endl;
    unsigned short Xi[3] = { 0, 0, y * y * y };
    for (int x = 0; x < w; x++) {
      int i = (h - y - 1) * w + x;
      for (int sy = 0; sy < 2; sy++) {
        for (int sx = 0; sx < 2; sx++) {
          r = Vec();
          for (int s = 0; s < samps; s++) {
            double rx = 2 * erand48(Xi), dx = rx < 1 ? sqrt(rx) - 1 : 1 - sqrt(2 - rx);
            double ry = 2 * erand48(Xi), dy = ry < 1 ? sqrt(ry) - 1 : 1 - sqrt(2 - ry);
            Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5)
              + cy * (((sy + .5 + dy) / 2 + y) / h - .5)
              + cam.d;
            r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1. / samps);
          }
          c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
        }
      }
    }
  }

  unsigned char *buffer = (unsigned char*)malloc(w * h * 3);
  const int R = 0, G = w * h, B = w * h * 2;
  for (int i = 0; i < w * h; i++) {
    buffer[i + R] = toInt(c[i].x);
    buffer[i + G] = toInt(c[i].y);
    buffer[i + B] = toInt(c[i].z);
  }
  CImg<unsigned char> img(buffer, w, h, 1, 3, true);
  img.save_png("output-test.png");
  free(buffer);
}
