#ifndef HITABLE_LIST_H__
#define HITABLE_LIST_H__

#include "AABB.h"
#include "Hitable.h"

class HitableList : public Hitable {
 public:
  Hitable** list;
  int size;

  HitableList() {}
  HitableList(Hitable** list, int size) : list(list), size(size) {}
  ~HitableList() {}
  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const;
  virtual bool bounding_box(float t0, float t1, AABB& box) const;
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

bool HitableList::bounding_box(float t0, float t1, AABB& box) const {
  if (size < 1) return false;
  AABB temp_box;
  bool first_true = list[0]->bounding_box(t0, t1, temp_box);
  if (!first_true) {
    return false;
  } else {
    box = temp_box;
  }
  for (int i = 0; i < size; i++) {
    if (list[0]->bounding_box(t0, t1, temp_box)) {
      box = surrounding_box(box, temp_box);
    } else {
      return false;
    }
  }
  return true;
}

#endif  // !HITABLE_LIST_H__
