#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kinectfusion::image_proc {

template <typename PixelT>
class Image {
 public:
  using value_type = PixelT;

  Image() = default;
  Image(unsigned int width, unsigned int height)
      : width_(width), height_(height) {
    data_.resize(std::size_t{width} * height);
  }

  [[nodiscard]] const PixelT& at(unsigned int x, unsigned int y) const {
    return data_[(y * width_) + x];
  }
  [[nodiscard]] PixelT& at(unsigned int x, unsigned int y) {
    return data_[(y * width_) + x];
  }

  [[nodiscard]] unsigned int width() const { return width_; }
  [[nodiscard]] unsigned int height() const { return height_; }

  [[nodiscard]] std::vector<PixelT>& data() { return data_; }
  [[nodiscard]] const std::vector<PixelT>& data() const { return data_; }

 private:
  unsigned int width_{};
  unsigned int height_{};
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

}  // namespace kinectfusion::image_proc

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP */
