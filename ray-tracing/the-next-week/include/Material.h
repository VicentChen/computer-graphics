#ifndef MATERIAL_H__
#define MATERIAL_H__

#include "Common.h"
#include "Hitable.h"
#include "Ray.h"

Vec3 random_in_unit_sphere() {
  Vec3 p;
  do {
    p = 2.0 * Vec3(drand48(), drand48(), drand48()) - Vec3(1, 1, 1);
  } while (p.length() >= 1.0);
  return p;
}

Vec3 reflect(const Vec3& v, const Vec3& n) { return v - 2 * dot(v, n) * n; }

bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted) {
  Vec3 uv = unit_vector(v);
  float cos_i = dot(uv, n);
  float cos_t2 = 1.0 - ni_over_nt * ni_over_nt * (1.0 - cos_i * cos_i);
  if (cos_t2 > 0) {
    refracted = ni_over_nt * (uv - n * cos_i) - n * sqrt(cos_t2);
    return true;
  } else {
    return false;
  }
}

float schlick(float cosine, float ior) {
  float r0 = (1 - ior) / (1 + ior);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow((1 - cosine), 5);
}

class Material {
 public:
  virtual bool scatter(const Ray& ray_in, const HitRecord& rec, Vec3& attenuation,
                       Ray& scattered) const = 0;
};

class Lambertian : public Material {
 public:
  Vec3 albedo;

  Lambertian() {}
  Lambertian(Vec3 albedo) : albedo(albedo) {}
  ~Lambertian() {}

  virtual bool scatter(const Ray& ray_in, const HitRecord& rec, Vec3& attenuation,
                       Ray& scattered) const;
};

bool Lambertian::scatter(const Ray& ray_in, const HitRecord& rec, Vec3& attenuation,
                         Ray& scattered) const {
  Vec3 target = rec.p + rec.n + random_in_unit_sphere();
  scattered = Ray(rec.p, target - rec.p, ray_in.time());
  attenuation = albedo;
  return true;
}

class Metal : public Material {
 public:
  Vec3 albedo;
  float fuzz;

  Metal() {}
  Metal(Vec3 albedo, float f) : albedo(albedo) { fuzz = (f < 1.0) ? f : 1; }
  ~Metal() {}

  virtual bool scatter(const Ray& ray_in, const HitRecord& rec, Vec3& attenuation,
                       Ray& scattered) const;
};

bool Metal::scatter(const Ray& ray_in, const HitRecord& rec, Vec3& attenuation,
                    Ray& scattered) const {
  Vec3 reflected = reflect(unit_vector(ray_in.d), rec.n);
  scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere());
  attenuation = albedo;
  return (dot(scattered.d, rec.n) > 0);
}

class Dielectric : public Material {
 public:
  float ior;  // index of refraction

  Dielectric() {}
  Dielectric(float ior) : ior(ior) {}
  ~Dielectric() {}

  virtual bool scatter(const Ray& ray_in, const HitRecord& rec, Vec3& attenuation,
                       Ray& scattered) const;
};

bool Dielectric::scatter(const Ray& ray_in, const HitRecord& rec, Vec3& attenuation,
                        Ray& scattered) const {
  Vec3 out_normal;
  Vec3 reflected = reflect(ray_in.d, rec.n);
  Vec3 refracted;
  float ni_over_nt;
  float reflect_prob;
  float cosine;

  attenuation = Vec3(1.0, 1.0, 1.0);

  if (dot(rec.n, ray_in.d) > 0) {
    out_normal = -1 * rec.n;
    ni_over_nt = ior;
    cosine = dot(ray_in.d, rec.n) / ray_in.d.length();
    cosine = sqrt(1 - ior * ior * (1 - cosine * cosine));
  } else {
    out_normal = rec.n;
    ni_over_nt = 1.0 / ior;
    cosine = -dot(ray_in.d, rec.n) / ray_in.d.length();
  }

  if (refract(ray_in.d, out_normal, ni_over_nt, refracted)) {
    reflect_prob = schlick(cosine, ior);
  } else {
    reflect_prob = 1.0;
  }

  if (drand48() < reflect_prob) {
    scattered = Ray(rec.p, reflected);
  } else {
    scattered = Ray(rec.p, refracted);
  }
  return true;
}

#endif  // !MATERIAL_H__