#ifndef AABB_H__
#define AABB_H__

#include <cmath>
#include "Ray.h"
#include "Vec3.h"

inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }

class AABB {
 public:
  AABB() {}
  AABB(const Vec3& a, const Vec3& b) : _min(a), _max(b) {}

  Vec3 min() const { return _min; }
  Vec3 max() const { return _max; }

  bool hit(const Ray& r, float tmin, float tmax) const {
    for (int i = 0; i < 3; i++) {
      float invD = 1.0f / (r.d)[i];
      float t0 = (min()[i] - (r.o)[i]) * invD;
      float t1 = (max()[i] - (r.o)[i]) * invD;
      if (invD < 0.0f) {
        std::swap(t0, t1);
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax <= tmin) return false;
      }
    }
    return true;
  }

  Vec3 _min, _max;
};

AABB surrounding_box(AABB box0, AABB box1) {
  Vec3 small(fmin(box0.min().x(), box1.min().x()), fmin(box0.min().y(), box1.min().y()),
             fmin(box0.min().z(), box1.min().z()));
  Vec3 big(fmax(box0.max().x(), box1.max().x()), fmax(box0.max().y(), box1.max().y()),
           fmax(box0.max().z(), box1.max().z()));
  return AABB(small, big);
}

#endif  // !AABB_H__
