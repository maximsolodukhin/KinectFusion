#include <spng.h>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <kinectfusion/image_proc/write_png.hpp>
#include <memory>
#include <stdexcept>
#include <string>

namespace kinectfusion::image_proc {
namespace {

[[nodiscard]] std::string str_of(int error) { return {spng_strerror(error)}; }

void check(int error, const std::string& what) {
  if (error != SPNG_OK) {
    throw std::runtime_error(what + ": " + str_of(error));
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
        "Failed to configure PNG encoder output");

  spng_ihdr ihdr{};
  ihdr.width = image.width();
  ihdr.height = image.height();
  ihdr.bit_depth = 8;
  ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR_ALPHA;
  check(spng_set_ihdr(ctx.get(), &ihdr), "Failed to set PNG header");

  const std::size_t buffer_size = image.data().size() * sizeof(ColorImage::value_type);
  check(spng_encode_image(ctx.get(), image.data().data(), buffer_size,
                          SPNG_FMT_PNG, SPNG_ENCODE_FINALIZE),
        "Failed to encode PNG image");

  std::size_t png_size = 0;
  void* png_data = spng_get_png_buffer(ctx.get(), &png_size, nullptr);
  if (png_data == nullptr) {
    throw std::runtime_error("Failed to get encoded PNG buffer");
  }
  const std::unique_ptr<void, decltype(&std::free)> png_buffer{png_data,
                                                              &std::free};

  std::ofstream output{filename, std::ios::binary};
  if (!output) {
    throw std::runtime_error{"Failed to open PNG output: " + filename};
  }
  output.write(static_cast<const char*>(png_buffer.get()),
               static_cast<std::streamsize>(png_size));
}

}  // namespace kinectfusion::image_proc
