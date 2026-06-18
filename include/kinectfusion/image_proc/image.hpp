#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kinectfusion::image_proc {

template <typename PixelT>
class Image {
 public:
  using value_type = PixelT;

  Image() = default;
  Image(std::size_t width, std::size_t height)
      : width_(width), height_(height) {
    data_.resize(width * height);
  }
  Image(std::size_t width, std::size_t height, const PixelT& init)
      : width_(width), height_(height), data_(width * height, init) {}

  [[nodiscard]] const PixelT& at(std::size_t x, std::size_t y) const {
    return data_[(y * width_) + x];
  }
  [[nodiscard]] PixelT& at(std::size_t x, std::size_t y) {
    return data_[(y * width_) + x];
  }

  [[nodiscard]] std::size_t width() const { return width_; }
  [[nodiscard]] std::size_t height() const { return height_; }

  [[nodiscard]] std::vector<PixelT>& data() { return data_; }
  [[nodiscard]] const std::vector<PixelT>& data() const { return data_; }

 private:
  std::size_t width_{};
  std::size_t height_{};
  std::vector<PixelT> data_;
};

class DepthImage final : public Image<std::uint16_t> {
 public:
  using Image<std::uint16_t>::Image;
};

class ColorImage final : public Image<std::uint32_t> {
 public:
  using Image<std::uint32_t>::Image;
};

using Vector3fImage = Image<Eigen::Vector3f>;

}  // namespace kinectfusion::image_proc

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP */
