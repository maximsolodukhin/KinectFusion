#include <spng.h>

#include <climits>
#include <cstddef>
#include <format>
#include <kinectfusion/image_proc/file.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/image_proc/read_png.hpp>
#include <memory>
#include <stdexcept>
#include <string>

namespace kinectfusion::image_proc {

// per-type decode parameters
template <class T>
struct PngFormat;

template <>
struct PngFormat<DepthImage> {
  // exact copy: the file must be 16-bit grayscale, endian mess
  static constexpr spng_format fmt = SPNG_FMT_PNG;
  static constexpr bool verbatim = true;
};

template <>
struct PngFormat<ColorImage> {
  // spng converts any input to rgba8
  static constexpr spng_format fmt = SPNG_FMT_RGBA8;
  static constexpr bool verbatim = false;
};

std::string str_of(int error) { return {spng_strerror(error)}; }

void check(int error, const std::string& what) {
  if (error != SPNG_OK) {
    throw std::runtime_error(what + ": " + str_of(error));
  }
}

// verbatim decodes copy pixels as-is, so the file itself must be
// single-channel with exactly the pixel's bit width
void ensure_grayscale(const spng_ihdr& ihdr, std::size_t bit_depth,
                      const std::string& filename) {
  if (ihdr.color_type != SPNG_COLOR_TYPE_GRAYSCALE ||
      ihdr.bit_depth != bit_depth) {
    auto dstr = std::to_string(bit_depth);
    auto what =
        std::format("Expected {}-bit grayscale png: {}", dstr, filename);
    throw std::runtime_error(what);
  }
}

template <class T>
T read_png(const std::string& filename) {
  constexpr auto fmt = PngFormat<T>::fmt;
  constexpr auto verbatim = PngFormat<T>::verbatim;

  auto png_data = read_file(filename);
  // hacky way to call spng_ctx_free when ctx goes out of scope(self made
  // custom destructor xD)
  using spng_ctx_destr = decltype(&spng_ctx_free);
  const std::unique_ptr<spng_ctx, spng_ctx_destr> ctx{spng_ctx_new(0),
                                                      &spng_ctx_free};
  if (!ctx) {
    throw std::runtime_error("Failed to allocate spng context");
  }

  check(spng_set_png_buffer(ctx.get(), png_data.data(), png_data.size()),
        "Failed to set PNG buffer");

  spng_ihdr ihdr{};
  check(spng_get_ihdr(ctx.get(), &ihdr), "Failed to read PNG header");

  constexpr std::size_t type_size = sizeof(typename T::value_type);
  if constexpr (verbatim) {
    ensure_grayscale(ihdr, type_size * CHAR_BIT, filename);
  }

  std::size_t decoded_size = 0;

  check(spng_decoded_image_size(ctx.get(), fmt, &decoded_size),
        "Failed to query decoded size");

  T image(ihdr.width, ihdr.height);
  const std::size_t buffer_size = image.data().size() * type_size;
  if (decoded_size != buffer_size) {
    throw std::runtime_error("PNG pixel format does not match image type: " +
                             filename);
  }

  check(spng_decode_image(ctx.get(), image.data().data(), buffer_size, fmt, 0),
        "Failed to decode PNG data");

  return image;
}

template DepthImage read_png<DepthImage>(const std::string& filename);
template ColorImage read_png<ColorImage>(const std::string& filename);

}  // namespace kinectfusion::image_proc
