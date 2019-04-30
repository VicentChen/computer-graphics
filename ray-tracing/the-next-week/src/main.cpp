#include <cfloat>
#include <iostream>
#include <svpng.inc>
#include "Camera.h"
#include "Common.h"
#include "HitableList.h"
#include "Material.h"
#include "Sphere.h"

using namespace std;

Vec3 color(const Ray& r, HitableList* world, int depth) {
  HitRecord rec;
  if (world->hit(r, 0.001, FLT_MAX, rec)) {
    Ray scattered;
    Vec3 attenuation;
    if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
      return attenuation * color(scattered, world, depth + 1);
    } else {
      return Vec3(0, 0, 0);
    }
  } else {
    // background
    Vec3 unit_direction = unit_vector(r.d);
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
  }
}

int main(int argc, char* argv) {
  // width,height,samples,channels
  const int W = 512, H = 512, S = 16, C = 3;

  unsigned char* img = new unsigned char[W * H * C];

  Vec3 origin;  // origin at (0, 0, 0)
  Vec3 horizontal(2.0, 0.0, 0.0);
  Vec3 vertical(0.0, 2.0, 0.0);
  Vec3 lower_left_corner(-1.0, -1.0, -1.0);

  Hitable* list[5];
  list[0] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.1, 0.2, 0.5)));
  list[1] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
  list[2] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0));
  list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5));
  list[4] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5));

  HitableList* world = new HitableList(list, 5);

  Vec3 lookfrom(5, 5, 5);
  Vec3 lookat(0, 0, -1);
  float dist_to_fcus = 10.0;
  float aperture = 0.1;
  Camera camera(lookfrom, lookat, Vec3(0, 1, 0), 20, W * 1.0 / H, aperture, dist_to_fcus);

  Vec3 c;
  const int R = 0, G = 1, B = 2;
#pragma omp parallel for schedule(dynamic, 1) private(c)  // OpenMP
  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      c = Vec3();
      for (int s = 0; s < S; s++) {
        float u = (w + drand48()) / W;
        float v = (h + drand48()) / H;
        Ray ray = camera.get_ray(u, v);
        c += color(ray, world, 0) / S;
      }

      unsigned char* pixel = img + ((H - h - 1) * W + w) * C;
      pixel[R] = (unsigned char)floor(sqrtf(c.r()) * 255 + .5);
      pixel[G] = (unsigned char)floor(sqrtf(c.g()) * 255 + .5);
      pixel[B] = (unsigned char)floor(sqrtf(c.b()) * 255 + .5);
    }
  }

  SaveImage("../../doc/img/the-next-week/test.png", img, W, H, C);

  return 0;
}
