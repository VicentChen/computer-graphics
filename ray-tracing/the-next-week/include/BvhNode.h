#ifndef BVH_NODE_H__
#define BVH_NODE_H__

#include "AABB.h"
#include "Common.h"
#include "Hitable.h"

class BvhNode : public Hitable {
 public:
  BvhNode() {}
  BvhNode(Hitable** l, int n, float t0, float t1);
  virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const;
  virtual bool bounding_box(float t0, float t1, AABB& box) const;

  Hitable* left;
  Hitable* right;
  AABB box;
};

int box_x_compare(const void* a, const void* b) {
  AABB box_left, box_right;
  Hitable* ah = *(Hitable**)a;
  Hitable* bh = *(Hitable**)b;
  if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
    std::cerr << "no bounding box in BvhNode constructor\n";
  if (box_left.min().x() - box_right.min().x() < 0.0)
    return -1;
  else
    return 1;
}

int box_y_compare(const void* a, const void* b) {
  AABB box_left, box_right;
  Hitable* ah = *(Hitable**)a;
  Hitable* bh = *(Hitable**)b;
  if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
    std::cerr << "no bounding box in BvhNode constructor\n";
  if (box_left.min().y() - box_right.min().y() < 0.0)
    return -1;
  else
    return 1;
}

int box_z_compare(const void* a, const void* b) {
  AABB box_left, box_right;
  Hitable* ah = *(Hitable**)a;
  Hitable* bh = *(Hitable**)b;
  if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
    std::cerr << "no bounding box in BvhNode constructor\n";
  if (box_left.min().z() - box_right.min().z() < 0.0)
    return -1;
  else
    return 1;
}

BvhNode::BvhNode(Hitable** l, int n, float t0, float t1) {
  int axis = int(3 * drand48());
  if (axis == 0)
    qsort(l, n, sizeof(Hitable*), box_x_compare);
  else if (axis == 1)
    qsort(l, n, sizeof(Hitable*), box_y_compare);
  else
    qsort(l, n, sizeof(Hitable*), box_z_compare);

  if (n == 1) {
    left = right = l[0];
  } else if (n == 2) {
    left = l[0];
    right = l[1];
  } else {
    left = new BvhNode(l, n / 2, t0, t1);
    right = new BvhNode(l + n / 2, n - n / 2, t0, t1);
  }
  
  AABB box_left, box_right;
  if (!left->bounding_box(t0, t1, box_left) || !right->bounding_box(t0, t1, box_right))
    std::cerr << "no bounding box in BvhNode constructor\n";
  box = surrounding_box(box_left, box_right);
}

bool BvhNode::hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const {
  if (box.hit(ray, t_min, t_max)) {
    HitRecord left_rec, right_rec;
    bool hit_left = left->hit(ray, t_min, t_max, left_rec);
    bool hit_right = right->hit(ray, t_min, t_max, right_rec);
    if (hit_left && hit_right) {
      if (left_rec.t < right_rec.t)
        rec = left_rec;
      else
        rec = right_rec;
      return true;
    } else if (hit_left) {
      rec = left_rec;
      return true;
    } else if (hit_right) {
      rec = right_rec;
      return true;
    } else {
      return false;
    }
  } else
    return false;
}

bool BvhNode::bounding_box(float t0, float t1, AABB& box) const {
  box = this->box;
  return true;
}

#endif  // !BVH_NODE_H__
