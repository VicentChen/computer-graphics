#ifndef VEC3_H__
#define VEC3_H__

#include <cmath>
#include <iostream>

class Vec3 {
 public:
  float e[3];

  Vec3(float e0 = .0, float e1 = .0, float e2 = .0) {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
  }
  ~Vec3() {}

  inline float x() { return e[0]; }
  inline float y() { return e[1]; }
  inline float z() { return e[2]; }
  inline float r() { return e[0]; }
  inline float g() { return e[1]; }
  inline float b() { return e[2]; }

  inline const Vec3& operator+() const { return *this; }
  inline Vec3 operator~() const { return Vec3(-e[0], -e[1], -e[2]); }
  inline float operator[](int i) const { return e[i]; }
  inline float& operator[](int i) { return e[i]; }

  inline Vec3& operator+=(const Vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }
  inline Vec3& operator-=(const Vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
  }
  inline Vec3& operator*=(const Vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
  }
  inline Vec3& operator/=(const Vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
  }
  inline Vec3& operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }
  inline Vec3& operator/=(const float t) {
    e[0] /= t;
    e[1] /= t;
    e[2] /= t;
    return *this;
  }

  inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
  inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
  inline void make_unit_vector() { *this *= 1 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
};

inline std::istream& operator>>(std::istream& is, Vec3& v) {
  is >> v.e[0] >> v.e[1] >> v.e[2];
  return is;
}
inline std::ostream& operator<<(std::ostream& os, Vec3& v) {
  os << v.e[0] << v.e[1] << v.e[2];
  return os;
}

inline Vec3 operator+(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}
inline Vec3 operator-(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}
inline Vec3 operator*(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}
inline Vec3 operator/(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline Vec3 operator*(float t, const Vec3& v) { return Vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t); }
inline Vec3 operator*(const Vec3& v, float t) { return Vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t); }
inline Vec3 operator/(const Vec3& v, float t) { return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t); }

inline float dot(const Vec3& v1, const Vec3& v2) {
  return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}
inline Vec3 cross(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.e[1] * v2.e[2] - v2.e[1] * v1.e[2], v1.e[2] * v2.e[0] - v2.e[2] * v1.e[0],
              v1.e[0] * v2.e[1] - v2.e[0] * v1.e[1]);
}
inline Vec3 unit_vector(Vec3 v) { return v / v.length(); }

#endif  // !VEC3_H__