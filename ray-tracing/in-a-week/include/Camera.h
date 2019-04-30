#ifndef CAMREA_H__
#define CAMERA_H__

#include <cmath>
#include "Common.h"
#include "Ray.h"
#include "Vec3.h"

Vec3 random_in_unit_disk() {
  Vec3 p;
  do {
    p = 2.0 * Vec3(drand48(), drand48(), 0) - Vec3(1, 1, 0);
  } while (dot(p, p) >= 1.0);
  return p;
}

class Camera {
 public:
  Vec3 origin;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 lower_left_corner;
  Vec3 u, v, w;
  float lens_radius;

  Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect, float aperture,
         float focus_dist) {
    lens_radius = aperture / 2;
    float theta = vfov * M_PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    origin = lookfrom;

    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = unit_vector(cross(w, u));

    horizontal = 2.0 * half_width * focus_dist * u;
    vertical = 2.0 * half_height * focus_dist * v;
    lower_left_corner = Vec3(-half_width, -half_height, -1.0);
    lower_left_corner =
        origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
  }
  ~Camera() {}
  Ray get_ray(float u, float v) {
    Vec3 rd = lens_radius * random_in_unit_disk();
    Vec3 offset = u * rd.x() + v * rd.y();
    return Ray(origin + offset,
               lower_left_corner + u * horizontal + v * vertical - origin - offset);
  }
};

#endif  // !CAMREA_H__
