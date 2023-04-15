// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

// Eigen still uses the __CUDACC_VER__ macro,
// which is deprecated in CUDA 9.
#if __CUDACC_VER_MAJOR__ >= 9
#undef __CUDACC_VER__
#define __CUDACC_VER__ \
  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif

#include <Eigen/Core>
#include <Eigen/LU>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "tsdfvh/tsdf_volume.h"
#include "utils/rgbd_sensor.h"
#include "marching_cubes/mesh.h"
#include "Logger.h"

namespace refusion {

	typedef Eigen::Matrix<double, 6, 1> Vector6d;

/**
 * @brief      Options for the tracker (see Tracker)
 */
	struct TrackerOptions
	{
		/** Maximum number of iteration per subsampling level **/
		int max_iterations_per_level[3];

		/** Downsampling for each level **/
		int downsample[3];

		/** Minimum norm of the increment to terminate the least-squares algorithm **/
		float min_increment;

		/** Constant used by the Huber estimator **/
		float huber_constant;

		/** Initial regularization term of Levenberg-Marquardt **/
		float regularization;

		/** Whether to remove dynamic elements from the model **/
		string reconstruction_strategy;

		/** Whether to use output the mask used for dynamic removal **/
		bool output_mask_video;
	};

/**
 * @brief      Class that tracks the RGB-D sensor and maintains the TSDF volume.
 */
	class Tracker
	{
	public:
		/**
		 * @brief      Constructs the object.
		 *
		 * @param[in]  tsdf_options     The options for the TSDF representation
		 * @param[in]  tracker_options  The options for the tracking algorithm
		 * @param[in]  sensor           The intrinsic parameters of the sensor
		 */
		Tracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
				const TrackerOptions &tracker_options, const RgbdSensor &sensor, Logger *logger);

		/**
		 * @brief      Destroys the object.
		 */
		~Tracker();

		/**
		 * @brief      Registers a new scan w.r.t. the model and integrates it into
		 *             the map.
		 *
		 * @param[in]  rgb    The RGB image
		 * @param[in]  depth  The depth image
		 */
		virtual void AddScan(const cv::Mat &rgb, const cv::Mat &depth) = 0;

		/**
		 * @brief      Gets the current pose of the sensor.
		 *
		 * @return     The current pose of the sensor.
		 */
		Eigen::Matrix4d GetCurrentPose();

		/**
		 * @brief      Extracts a mesh from the TSDF volume delimited by a specified
		 *             bounding box.
		 *
		 * @param[in]  lower_corner  The lower corner of the bounding box
		 * @param[in]  upper_corner  The upper corner of the bounding box
		 *
		 * @return     The mesh.
		 */
		tsdfvh::Mesh ExtractMesh(const float3 &lower_corner,
								 const float3 &upper_corner);

		/**
		 * @brief      Generates a virtual RGB image from the model.
		 *
		 * @param[in]  width   The desired width of the image
		 * @param[in]  height  The desired height of the image
		 *
		 * @return     The virtual RGB image.
		 */
		cv::Mat GenerateRgb(int width, int height);

	protected:
		/** Write the mask to the file */
		void LogMask(bool *mask, RgbdImage &image);

		/** TSDF volume */
		tsdfvh::TsdfVolume *volume_;

		/** Options for the tracker */
		TrackerOptions options_;

		/** Intrinsic parameters of the RGB-D sensor */
		RgbdSensor sensor_;

		/** Current pose of the sensor */
		Eigen::Matrix4d pose_;

		/** Previous computed increment */
		Vector6d prev_increment_;

		/** Used for directly integrating the first scan in the model without pose
		 *  estimation
		 */
		bool first_scan_ = true;

		/** Used for logging */
		Logger *logger_;
	};

	/**
	 * @brief      Implementation of a tracker which performs the reconstruction as described in ReFusion paper.
	 */
	class ReTracker : public refusion::Tracker
	{
	public:
		ReTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
				  const refusion::TrackerOptions &tracker_options, const refusion::RgbdSensor &sensor, Logger *logger);

		~ReTracker() = default;

		void AddScan(const cv::Mat &rgb, const cv::Mat &depth) override;

	protected:
		/**
		 * @brief      Estimates the pose of the sensor.
		 *
		 * @param[in]  image        The RGB-D image
		 * @param      mask         The mask containing dynamic elements. It is
		 *                          created if create_mask is set to true
		 * @param[in]  create_mask  If true, mask will be created
		 */
		void TrackCamera(const refusion::RgbdImage &image, bool *mask, bool create_mask);
	};


	/**
	 * @brief      Modifiation of the base tracker provided in source code for refusion to perform static
	 * 			reconstruction; all the dynamic removal stuff has been removed here. To make a baseline.
	 */
	class StaticTracker : public refusion::Tracker
	{
	public:
		StaticTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
					  const refusion::TrackerOptions &tracker_options, const refusion::RgbdSensor &sensor, Logger *logger);

		~StaticTracker() = default;

		void AddScan(const cv::Mat &rgb, const cv::Mat &depth) override;
	protected:
		void TrackCamera(const refusion::RgbdImage &image, bool *mask);
	};


	/**
	 * @brief  Novel tracker implementation which uses Optical Flow to create a mask and remove dynamic objects.
	 */
	class OpticalFlowTracker : public refusion::Tracker
	{
	public:
		OpticalFlowTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
						  const refusion::TrackerOptions &tracker_options, const refusion::RgbdSensor &sensor, Logger *logger);

		~OpticalFlowTracker() = default;

		void AddScan(const cv::Mat &rgb, const cv::Mat &depth) override;
	protected:
		void TrackCamera(const refusion::RgbdImage &image, bool *mask);
		cv::Mat prev_rgb_frame;
		cv::Mat prev_depth_frame;
	};

	refusion::Tracker *CreateTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
					   const TrackerOptions &tracker_options, const RgbdSensor &sensor, Logger *logger);

}  // namespace refusion
