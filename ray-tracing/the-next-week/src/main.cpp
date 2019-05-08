#include <cfloat>
#include <iostream>
#include <svpng.inc>
#include "BvhNode.h"
#include "Camera.h"
#include "Common.h"
#include "HitableList.h"
#include "Material.h"
#include "MovingSphere.h"
#include "Sphere.h"
#include "Texture.h"

using namespace std;

Vec3 color(const Ray& r, Hitable* world, int depth) {
  HitRecord rec;
  if (world->hit(r, 0.001, FLT_MAX, rec)) {
    Ray scattered;
    Vec3 attenuation;
    Vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
    if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
      return emitted + attenuation * color(scattered, world, depth + 1);
    } else {
      return emitted;
    }
  } else {
    return Vec3(0, 0, 0);
  }
}

Hitable* random_scene() {
  int n = 500;

  Hitable** list = new Hitable*[n + 1];
  Texture* checker = new CheckerTexture(new ConstantTexture(Vec3(0.2, 0.3, 0.1)),
                                      new ConstantTexture(Vec3(0.9, 0.9, 0.9)));
  list[0] = new Sphere(Vec3(0, -1000, 0), 1000, new Lambertian(checker));
  int i = 1;

  for (int a = -5; a < 5; a++) {
    for (int b = -5; b < 5; b++) {
      float choose_mat = drand48();

      Vec3 center(a + 0.9 * drand48(), 0.2, b + 0.9 * drand48());

      if ((center - Vec3(4, 0.2, 0)).length() > 0.9) {
        if (choose_mat < 0.8) {  // diffuse
          list[i++] = new MovingSphere(
              center, center + Vec3(0, 0.5 * drand48(), 0), 0.0, 1.0, 0.2,
              new Lambertian(new ConstantTexture(
                  Vec3(drand48() * drand48(), drand48() * drand48(), drand48() * drand48()))));
        } else if (choose_mat < 0.95) {  // metal
          list[i++] = new Sphere(
              center, 0.2,
              new Metal(Vec3(0.5 * (1 + drand48()), 0.5 * (1 + drand48()), 0.5 * (1 + drand48())),
                        0.5 * drand48()));
        } else {  // glass
          list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
        }
      }
    }
  }

  list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5));
  list[i++] =
      new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(new ConstantTexture(Vec3(0.4, 0.2, 0.1))));
  list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

  // return new HitableList(list, i);
  return new BvhNode(list, i, 0.0, 1.0);
}

Hitable* earth() {
  int W, H, C;
  // TODO: memory leaks here
  unsigned char* img = LoadImage("../dependencies/img/earth.jpg", &W, &H, &C);
  Material* earth = new Lambertian(new ImageTexture(img, W, H));
  return new Sphere(Vec3(0, 0, 0), 2, earth);
}

Hitable* two_spheres() {
  Texture* checker = new CheckerTexture(new ConstantTexture(Vec3(0.2, 0.3, 0.1)),
                                        new ConstantTexture(Vec3(0.9, 0.9, 0.9)));
  int n = 50;
  Hitable** list = new Hitable*[n + 1];
  list[0] = new Sphere(Vec3(0, -10, 0), 10, new Lambertian(checker));
  list[1] = new Sphere(Vec3(0,  10, 0), 10, new Lambertian(checker));
  return new HitableList(list, 2);
}

Hitable* two_perlin_spheres() {
  Texture* pertext = new NoiseTexture(4);
  int n = 50;
  Hitable** list = new Hitable*[n + 1];
  list[0] = new Sphere(Vec3(0, -1000, 0), 1000, new Lambertian(pertext));
  list[1] = new Sphere(Vec3(0, 2, 0), 2, new Lambertian(pertext));
  return new HitableList(list, 2);
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
  list[0] =
      new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(new ConstantTexture(Vec3(0.1, 0.2, 0.5))));
  list[1] = new Sphere(Vec3(0, -100.5, -1), 100,
                       new Lambertian(new ConstantTexture(Vec3(0.8, 0.8, 0.0))));
  list[2] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0));
  list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5));
  list[4] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5));

  Hitable* world = new HitableList(list, 5);
  //world = random_scene();
  //world = two_perlin_spheres();
  world = earth();

  Vec3 lookfrom(13, 2, 3);
  Vec3 lookat(0, 0, 0);
  float dist_to_fcus = 10.0;
  float aperture = 0.0;
  Camera camera(lookfrom, lookat, Vec3(0, 1, 0), 20, W * 1.0 / H, aperture, dist_to_fcus, 0.0, 1.0);

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

  SaveImage("../../doc/img/the-next-week/Texture.png", img, W, H, C);

  return 0;
}
