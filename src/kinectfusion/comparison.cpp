#include <cmath>
#include <cstddef>
#include <kinectfusion/comparison.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <span>

namespace kinectfusion {

VolumeComparison Comparator::compare(ConstHostVolumeView primary,
                                     ConstHostVolumeView reference) {
  require(primary.geometry == reference.geometry,
          "Volume comparison requires matching geometry");

  PairTally tally;
  Deviation distance;
  Deviation weight;

  const std::span our_voxels = primary.voxel_span();
  const std::span their_voxels = reference.voxel_span();

  for (std::size_t i = 0; i < our_voxels.size(); ++i) {
    const Voxel& ours = our_voxels[i];
    const Voxel& theirs = their_voxels[i];
    if (!tally.classify(ours.weight_value() > 0.0F,
                        theirs.weight_value() > 0.0F)) {
      continue;
    }
    distance.add(std::abs(ours.tsdf() - theirs.tsdf()));
    weight.add(std::abs(ours.weight_value() - theirs.weight_value()));
  }

  return VolumeComparison{.compared_voxels = tally.compared,
                          .only_primary = tally.only_primary,
                          .only_reference = tally.only_reference,
                          .max_distance_delta = distance.max(),
                          .mean_distance_delta = distance.mean(tally.compared),
                          .max_weight_delta = weight.max()};
}

SurfaceMapsComparison Comparator::compare(const SurfaceMaps& primary,
                                          const SurfaceMaps& reference) {
  const auto matches_shape = [&](const auto& image) {
    return image.width() == primary.points.width() &&
           image.height() == primary.points.height();
  };
  require(matches_shape(primary.normals) && matches_shape(reference.points) &&
              matches_shape(reference.normals),
          "Surface map comparison requires matching dimensions");

  PairTally tally;
  Deviation point_distance;
  Deviation normal_angle;

  const auto& our_points = primary.points.data();
  const auto& our_normals = primary.normals.data();

  const auto& their_points = reference.points.data();
  const auto& their_normals = reference.normals.data();

  for (std::size_t i = 0; i < our_points.size(); ++i) {
    if (!tally.classify(
            valid_surface_pixel(our_points.at(i), our_normals.at(i)),
            valid_surface_pixel(their_points.at(i), their_normals.at(i)))) {
      continue;
    }

    point_distance.add(norm(our_points.at(i) - their_points.at(i)));
    normal_angle.add(angle_between(our_normals.at(i), their_normals.at(i)));
  }
  return SurfaceMapsComparison{
      .compared_pixels = tally.compared,
      .only_primary = tally.only_primary,
      .only_reference = tally.only_reference,
      .max_point_distance = point_distance.max(),
      .mean_point_distance = point_distance.mean(tally.compared),
      .max_normal_angle = normal_angle.max(),
      .mean_normal_angle = normal_angle.mean(tally.compared)};
}

}  // namespace kinectfusion
