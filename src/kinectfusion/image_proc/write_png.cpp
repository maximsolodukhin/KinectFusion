#include <spng.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/image_proc/write_png.hpp>
#include <memory>
#include <stdexcept>
#include <string>

namespace kinectfusion::image_proc {
namespace {

constexpr std::uint8_t kPngBitDepth = 8;

void check(int error, const std::string& what) {
  if (error != SPNG_OK) {
    throw std::runtime_error(what + ": " + spng_strerror(error));
  }
}

}  // namespace

void write_png(const ColorImage& image, const std::string& filename) {
  using spng_ctx_destr = decltype(&spng_ctx_free);
  const std::unique_ptr<spng_ctx, spng_ctx_destr> ctx{
      spng_ctx_new(SPNG_CTX_ENCODER), &spng_ctx_free};
  if (!ctx) {
    throw std::runtime_error("Failed to allocate spng encoder context");
  }

  check(spng_set_option(ctx.get(), SPNG_ENCODE_TO_BUFFER, 1),
        "Failed to enable PNG encode-to-buffer");
  check(spng_set_option(ctx.get(), SPNG_IMG_COMPRESSION_LEVEL, 1),
        "Failed to set PNG compression level");
  check(spng_set_option(ctx.get(), SPNG_FILTER_CHOICE, SPNG_DISABLE_FILTERING),
        "Failed to set PNG filter choice");
  // Saves time, but produces larger files. The default is 6, which is slower
  // but smaller. Who cares? We're benchmarking the kinectfusion pipeline, not
  // spng library.

  spng_ihdr ihdr{};
  ihdr.width = static_cast<std::uint32_t>(image.width());
  ihdr.height = static_cast<std::uint32_t>(image.height());
  ihdr.bit_depth = kPngBitDepth;
  ihdr.color_type = static_cast<std::uint8_t>(SPNG_COLOR_TYPE_TRUECOLOR_ALPHA);

  check(spng_set_ihdr(ctx.get(), &ihdr), "Failed to set PNG header");

  const std::size_t length =
      image.data().size() * sizeof(ColorImage::value_type);
  check(spng_encode_image(ctx.get(), image.data().data(), length, SPNG_FMT_PNG,
                          SPNG_ENCODE_FINALIZE),
        "Failed to encode PNG image");

  int error = 0;
  std::size_t png_size = 0;
  void* png = spng_get_png_buffer(ctx.get(), &png_size, &error);
  check(error, "Failed to obtain encoded PNG buffer");
  const std::unique_ptr<void, decltype(&std::free)> png_guard{png, &std::free};

  std::ofstream output{filename, std::ios::binary};
  if (!output) {
    throw std::runtime_error("Failed to open PNG output: " + filename);
  }
  output.write(static_cast<const char*>(png),
               static_cast<std::streamsize>(png_size));
}

}  // namespace kinectfusion::image_proc
