#include <features/dspsift.h>

#include <iostream>
#include <vector>
#include <Eigen/Core>

extern "C" {
#include <time.h>
#include <vl/covdet.h>
#include <vl/sift.h>
}

struct FeatureKeypoint {
  FeatureKeypoint() : x(0), y(0), a11(1), a12(0), a21(0), a22(1) {}
  FeatureKeypoint(float x, float y, float scale, float orientation) : x(x_), y(y_) {
    const float scale_cos_orientation = scale * std::cos(orientation);
    const float scale_sin_orientation = scale * std::sin(orientation);
    a11 = scale_cos_orientation;
    a12 = -scale_sin_orientation;
    a21 = scale_sin_orientation;
    a22 = scale_cos_orientation;
  }

  /*
  FeatureKeypoint(float x, float y, float a11, float a12, float a21, float a22);


  // Rescale the feature location and shape size by the given scale factor.
  void Rescale(float scale);
  void Rescale(float scale_x, float scale_y);

  // Compute shape parameters from affine shape.
  float ComputeScale() const;
  float ComputeScaleX() const;
  float ComputeScaleY() const;
  float ComputeOrientation() const;
  float ComputeShear() const;
*/
  // Location of the feature, with the origin at the upper left image corner,
  // i.e. the upper left pixel has the coordinate (0.5, 0.5).
  float x;
  float y;

  // Affine shape of the feature.
  float a11;
  float a12;
  float a21;
  float a22;
};

using VlSiftType = std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)>;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptorsFloat;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptors;
typedef std::vector<FeatureKeypoint> FeatureKeypoints;

template <typename T1, typename T2>
T2 TruncateCast(const T1 value) {
  return static_cast<T2>(std::min(
      static_cast<T1>(std::numeric_limits<T2>::max()),
      std::max(static_cast<T1>(std::numeric_limits<T2>::min()), value)));
}

FeatureDescriptors FeatureDescriptorsToUnsignedByte(
    const Eigen::Ref<const FeatureDescriptorsFloat>& descriptors) {
  FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
                                               descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
          TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

namespace features {

py::tuple dspsift(foundation::pyarray_f image, float peak_threshold,
                float edge_threshold, int target_num_features) {
  if (!image.size()) {
    return py::none();
  }

  // std::vector<float> points_;
  // std::vector<float> desc_;
  vl_size numFeatures;

  int num_octaves = 4;
  int octave_resolution = 3;
  int first_octave = -1;
  bool feature_root = true;
  bool estimate_affine_shape = true;
  bool domain_size_pooling = true;

  double dsp_min_scale = 1.0 / 6.0;
  double dsp_max_scale = 3.0;
  int dsp_num_scales = 10;
  
  // TODO: allocation?
  FeatureDescriptors *descriptors = new FeatureDescriptors(); // TODO: use smart pointers
  FeatureKeypoints *keypoints = new FeatureKeypoints();

  {
    py::gil_scoped_release release;

    // Setup covariant SIFT detector.
    std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
        vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
    if (!covdet) {
      return false;
    }

    const int kMaxOctaveResolution = 1000;

    vl_covdet_set_first_octave(covdet.get(), first_octave);
    vl_covdet_set_octave_resolution(covdet.get(), octave_resolution);
    vl_covdet_set_peak_threshold(covdet.get(), peak_threshold);
    vl_covdet_set_edge_threshold(covdet.get(), edge_threshold);

    // TODO: does it need to be normalized [0-1]?
    vl_covdet_put_image(covdet, image.data(), image.shape(1), image.shape(0));
    

    // create a detector object
    VlCovDet *covdet = vl_covdet_new(VL_COVDET_METHOD_HESSIAN);

    // set various parameters (optional)
    vl_covdet_set_first_octave(covdet, 0);
    vl_covdet_set_peak_threshold(covdet, peak_threshold);
    vl_covdet_set_edge_threshold(covdet, edge_threshold);

    // process the image and run the detector
    vl_covdet_put_image(covdet, image.data(), image.shape(1), image.shape(0));
    // vl_covdet_set_non_extrema_suppression_threshold(covdet, 0);
    vl_covdet_detect(covdet, target_num_features);

    if (estimate_affine_shape){
      vl_covdet_extract_affine_shape(covdet.get());
    } else {
      vl_covdet_extract_orientations(covdet.get());
    }

    const int num_features = vl_covdet_get_num_features(covdet.get());
    VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

    // Sort features according to detected octave and scale.
    std::sort(
        features,
        features + num_features,
        [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
          if (feature1.o == feature2.o) {
            return feature1.s > feature2.s;
          } else {
            return feature1.o > feature2.o;
          }
        });

    const size_t max_num_features =
        static_cast<size_t>(target_num_features);

    // Copy detected keypoints and clamp when maximum number of features
    // reached.
    int prev_octave_scale_idx = std::numeric_limits<int>::max();
    for (int i = 0; i < num_features; ++i) {
      FeatureKeypoint keypoint;
      keypoint.x = features[i].frame.x + 0.5;
      keypoint.y = features[i].frame.y + 0.5;
      keypoint.a11 = features[i].frame.a11;
      keypoint.a12 = features[i].frame.a12;
      keypoint.a21 = features[i].frame.a21;
      keypoint.a22 = features[i].frame.a22;
      keypoints->push_back(keypoint);

      const int octave_scale_idx =
          features[i].o * kMaxOctaveResolution + features[i].s;

      if (octave_scale_idx != prev_octave_scale_idx &&
          keypoints->size() >= max_num_features) {
        break;
      }

      prev_octave_scale_idx = octave_scale_idx;
    }

    // Compute the descriptors for the detected keypoints.
    if (descriptors != nullptr) {
      descriptors->resize(keypoints->size(), 128);

      const size_t kPatchResolution = 15;
      const size_t kPatchSide = 2 * kPatchResolution + 1;
      const double kPatchRelativeExtent = 7.5;
      const double kPatchRelativeSmoothing = 1;
      const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
      const double kSigma =
          kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

      std::vector<float> patch(kPatchSide * kPatchSide);
      std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

      float d_min_scale = 1;
      float d_scale_step = 0;
      int d_num_scales = 1;
      if (domain_size_pooling) {
        d_min_scale = dsp_min_scale;
        d_scale_step = (dsp_max_scale - dsp_min_scale) /
                         dsp_num_scales;
        d_num_scales = dsp_num_scales;
      }

      FeatureDescriptorsFloat descriptor(1, 128);
      FeatureDescriptorsFloat scaled_descriptors(d_num_scales, 128);

      std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
          vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
      if (!sift) {
        return py::none();
      }

      vl_sift_set_magnif(sift.get(), 3.0);

      for (size_t i = 0; i < keypoints->size(); ++i) {
        for (int s = 0; s < d_num_scales; ++s) {
          const double dsp_scale = d_min_scale + s * d_scale_step;

          VlFrameOrientedEllipse scaled_frame = features[i].frame;
          scaled_frame.a11 *= dsp_scale;
          scaled_frame.a12 *= dsp_scale;
          scaled_frame.a21 *= dsp_scale;
          scaled_frame.a22 *= dsp_scale;

          vl_covdet_extract_patch_for_frame(covdet.get(),
                                            patch.data(),
                                            kPatchResolution,
                                            kPatchRelativeExtent,
                                            kPatchRelativeSmoothing,
                                            scaled_frame);

          vl_imgradient_polar_f(patchXY.data(),
                                patchXY.data() + 1,
                                2,
                                2 * kPatchSide,
                                patch.data(),
                                kPatchSide,
                                kPatchSide,
                                kPatchSide);

          vl_sift_calc_raw_descriptor(sift.get(),
                                      patchXY.data(),
                                      scaled_descriptors.row(s).data(),
                                      kPatchSide,
                                      kPatchSide,
                                      kPatchResolution,
                                      kPatchResolution,
                                      kSigma,
                                      0);
        }

        if (domain_size_pooling) {
          descriptor = scaled_descriptors.colwise().mean();
        } else {
          descriptor = scaled_descriptors;
        }

        if (!feature_root) {
          descriptors->rowwise().normalize();
        } else {
          for (Eigen::MatrixXf::Index r = 0; r < descriptors->rows(); ++r) {
            descriptors->row(r) *= 1 / descriptors->row(r).lpNorm<1>();
            descriptors->row(r) = descriptors->row(r).array().sqrt();
          }
        }

        descriptors->row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
      }

      // *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
    }
  }

  return py::make_tuple(
    foundation::py_array_from_data(keypoints.data(), keypoints->size(), 6),
    foundation::py_array_from_data(descriptors.data(), keypoints->size(), 128));
}

}  // namespace features
