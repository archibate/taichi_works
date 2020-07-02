#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

///////////////////////////////////////////////////////////////////////////////////
// https://stackoverflow.com/questions/11843226/multiplying-a-string-by-an-int-in-c
template<typename Char, typename Traits, typename Allocator>
std::basic_string<Char, Traits, Allocator> operator *
(const std::basic_string<Char, Traits, Allocator> s, size_t n)
{
   std::basic_string<Char, Traits, Allocator> tmp = s;
   for (size_t i = 0; i < n; ++i)
   {
      tmp += s;
   }
   return tmp;
}

template<typename Char, typename Traits, typename Allocator>
std::basic_string<Char, Traits, Allocator> operator *
(size_t n, const std::basic_string<Char, Traits, Allocator>& s)
{
   return s * n;
}
///////////////////////////////////////////////////////////////////////////////////

struct Particle {
  float pos;
};

struct BNode {
  std::unique_ptr<BNode> left, right;
  float total_mass{0};
  float total_pos{0};

  std::optional<int> chid;

  BNode() = default;

  BNode *touch_left() {
    if (!left)
      left = std::make_unique<BNode>();
    return left.get();
  }

  BNode *touch_right() {
    if (!right)
      right = std::make_unique<BNode>();
    return right.get();
  }

  bool is_leaf() {
    return left && right;
  }

  void add_point(int id, float pos) {
    if (!this->chid.has_value()) {
      *this->chid = id;
      return;
    }
    if (pos < 0.5)
      touch_left()->add_point(id, pos * 2);
    else
      touch_right()->add_point(id, pos * 2 - 1);
  }
};

struct Scene {
  std::vector<Particle> particles;
  std::unique_ptr<BNode> root;

  void init() {
    particles = std::vector<Particle>(1);
    for (int i = 0; i < particles.size(); i++) {
      particles[i].pos = float(rand()) / RAND_MAX;
    }
  }

  void run() {
    std::cout << "in!" << std::endl;
    root = std::make_unique<BNode>();
    for (int i = 0; i < particles.size(); i++) {
      std::cout << "add" << i << std::endl;
      root->add_point(i, particles[i].pos);
    }
  }
};


int main()
{
  Scene scene;
  scene.init();
  scene.run();
}
