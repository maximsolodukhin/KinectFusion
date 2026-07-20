// Pins the AppOptions -> library-config mapping: each user-facing option
// must land in the config structs. This mapping has silently regressed
// before, so each field is asserted against a non-default AppOptions.
#include <catch2/catch_test_macros.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/trilinear.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <variant>

#include "app_options.hpp"

namespace {

[[nodiscard]] app::AppOptions non_default_options() {
  app::AppOptions options;
  options.tsdf_variant = "classic";
  options.voxel_store = "bf16";
  options.color_store = "none";
  options.raycast_backend = "band-march";
  options.storage_layout = "sparse";
  options.sparse_capacity = 1234;
  options.integration_mode = "band";
  options.cell_gradient_normals = true;
  options.raycast_seed_previous = true;
  options.raycast_tsdf_from_valid_corners = true;
  options.icp_device_solve = true;
  options.icp_capture_graph = true;
  options.icp_damping = "diagonal";
  options.icp_lambda = 0.25F;
  options.icp_adaptive_damping = true;
  options.projective_tsdf_distance = false;
  options.distance_scaled_truncation = true;
  return options;
}

}  // namespace

TEST_CASE("Every CLI knob reaches the pipeline config", "[app_config]") {
  const auto options = non_default_options();
  const auto config = options.pipeline_config();

  CHECK(std::holds_alternative<kinectfusion::ClassicTsdf>(config.tsdf_rule));
  CHECK(config.voxel == kinectfusion::VoxelStore::kBf16);
  CHECK(config.color == kinectfusion::ColorStore::kNone);
  CHECK(config.raycast_backend == kinectfusion::RaycastBackend::kBandMarch);
  CHECK(config.storage == kinectfusion::StorageLayout::kSparse);
  CHECK(config.sparse_block_capacity == 1234);
  CHECK(config.integration.mode == kinectfusion::IntegrationMode::kBand);
  CHECK_FALSE(config.integration.projective_distance);
  CHECK(config.integration.distance_scaled_truncation);
  CHECK(config.raycast.cell_gradient_normals);
  CHECK(config.raycast.seed_from_previous);
  CHECK(config.raycast.tsdf_corner_policy ==
        kinectfusion::CornerPolicy::kRequireAll);
}

TEST_CASE("Every ICP knob reaches the tracker options", "[app_config]") {
  const auto options = non_default_options();
  const auto icp = options.icp_options();

  CHECK(icp.device_solve);
  CHECK(icp.device_graph_build == kinectfusion::IcpGraphBuild::kCaptured);
  CHECK(icp.damping.mode == kinectfusion::IcpDampingMode::kDiagonal);
  CHECK(icp.damping.lambda == 0.25F);
  CHECK(icp.adaptive_damping);
}
