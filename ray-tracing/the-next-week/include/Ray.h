#ifndef RAY_H__
#define RAY_H__

#include "Vec3.h"

class Ray {
 public:
  Vec3 o, d;

  Ray() {}
  Ray(const Vec3 &o, const Vec3 &d) : o(o), d(d) {}
  ~Ray() {}

  Vec3 origin() const { return o; }
  Vec3 direction() const { return d; }
  Vec3 point_at_parameter(float t) const { return o + t * d; }
};

#endif  // !RAY_H__
