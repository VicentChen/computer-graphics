#ifndef RAY_H__
#define RAY_H__

#include "Vec3.h"

class Ray {
 public:
  Ray() {}
  Ray(const Vec3 &o, const Vec3 &d, float t = .0) : o(o), d(d), _time(t) {}

  Vec3 origin() const { return o; }
  Vec3 direction() const { return d; }
  float time() const { return _time; }
  Vec3 point_at_parameter(float t) const { return o + t * d; }

  Vec3 o, d;
  float _time;
};

#endif  // !RAY_H__
