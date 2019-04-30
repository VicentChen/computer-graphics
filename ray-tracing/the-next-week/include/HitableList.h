#ifndef HITABLE_LIST_H__
#define HITABLE_LIST_H__

#include "Hitable.h"

class HitableList : public Hitable {
 public:
  Hitable** list;
  int size;

  HitableList() {}
  HitableList(Hitable** list, int size) : list(list), size(size) {}
  ~HitableList() {}
  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const;
};

bool HitableList::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
  HitRecord temp_rec;
  bool hit_anything = false;
  float closest_so_far = t_max;
  for (int i = 0; i < size; i++) {
    if (list[i]->hit(ray, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  return hit_anything;
}

#endif  // !HITABLE_LIST_H__
