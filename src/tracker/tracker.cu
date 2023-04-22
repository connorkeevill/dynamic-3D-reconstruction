// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tracker/tracker.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include "tracker/eigen_wrapper.h"
#include "utils/matrix_utils.h"
#include "utils/utils.h"
#include "utils/rgbd_image.h"
#include <stdlib.h>

#define THREADS_PER_BLOCK3 32

const float OPTICAL_FLOW_MOVEMENT_THRESHOLD = 3.5;

namespace refusion {
	refusion::Tracker *CreateTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
					 const TrackerOptions &tracker_options,
					 const RgbdSensor &sensor,
					 Logger *logger)
	{
		if(tracker_options.reconstruction_strategy == "residual")
		{
			return new refusion::ReTracker(tsdf_options, tracker_options, sensor, logger);
		}
		else if(tracker_options.reconstruction_strategy == "optical_flow")
		{
			return new refusion::OpticalFlowTracker(tsdf_options, tracker_options, sensor, logger);
		}
		else if(tracker_options.reconstruction_strategy == "static")
		{
			return new refusion::StaticTracker(tsdf_options, tracker_options, sensor, logger);
		}
		else
		{
			logger->alwaysLog("Unknown reconstruction strategy: " + tracker_options.reconstruction_strategy);
			logger->error("Reconstruction strategy not valid, exiting");
			exit(EXIT_FAILURE);
		}

		return nullptr;
	}

	Tracker::Tracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
					 const TrackerOptions &tracker_options,
					 const RgbdSensor &sensor,
					 Logger *logger)
	{
		cudaMallocManaged(&volume_, sizeof(tsdfvh::TsdfVolume));
		volume_->Init(tsdf_options);
		options_ = tracker_options;
		sensor_ = sensor;
		pose_ = Eigen::Matrix4d::Identity();
		logger_ = logger;
	}

	Tracker::~Tracker()
	{
		volume_->Free();
		cudaFree(volume_);
	}

	Eigen::Matrix4d v2t(const Vector6d &xi)
	{
		Eigen::Matrix4d M;

		M << 0.0, -xi(2), xi(1), xi(3),
				xi(2), 0.0, -xi(0), xi(4),
				-xi(1), xi(0), 0.0, xi(5),
				0.0, 0.0, 0.0, 0.0;

		return M;
	}

	float length(cv::Point2f p)
	{
		return sqrt(p.x * p.x + p.y * p.y);
	}

	cv::Point2f normalize(cv::Point2f p)
	{
		float len = length(p);
		return {p.x / len, p.y / len};
	}

	__host__ __device__ float Intensity(float3 color)
	{
		return 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
	}

	__host__ __device__ float ColorDifference(uchar3 c1, uchar3 c2)
	{
		float3 c1_float = ColorToFloat(c1);
		float3 c2_float = ColorToFloat(c2);
		return Intensity(c1_float) - Intensity(c2_float);
	}

	/**
	 * Used in camera tracking for ReTracker. This is terribly messy code, but there's too much linear algebra
	 * to clean it up right now.
	 * TODO: clean this up.
	 *
	 * @param volume
	 * @param huber_constant
	 * @param rgb
	 * @param depth
	 * @param mask
	 * @param transform
	 * @param sensor
	 * @param acc_H
	 * @param acc_b
	 * @param downsample
	 * @param residuals_threshold
	 * @param create_mask
	 */
	__global__ void CreateLinearSystemResidual(tsdfvh::TsdfVolume *volume,
									   float huber_constant, uchar3 *rgb,
									   float *depth, bool *mask, float4x4 transform,
									   RgbdSensor sensor, mat6x6 *acc_H,
									   mat6x1 *acc_b, int downsample,
									   float residuals_threshold,
									   bool create_mask)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int size = sensor.rows * sensor.cols;
		for (int idx = index; idx < size / (downsample * downsample); idx += stride) {
			mat6x6 new_H;
			mat6x1 new_b;
			new_H.setZero();
			new_b.setZero();
			int v = (idx / (sensor.cols / downsample)) * downsample;
			int u = (idx - (sensor.cols / downsample) * v / downsample) * downsample;
			int i = v * sensor.cols + u;
			if (depth[i] < volume->GetOptions().min_sensor_depth) {
				continue;
			}
			if (depth[i] > volume->GetOptions().max_sensor_depth) {
				continue;
			}
			float3 point = transform * GetPoint3d(i, depth[i], sensor);
			tsdfvh::Voxel v1 = volume->GetInterpolatedVoxel(point);
			if (v1.weight == 0) {
				continue;
			}
			float sdf = v1.sdf;
			float3 color = make_float3(static_cast<float>(v1.color.x) / 255,
									   static_cast<float>(v1.color.y) / 255,
									   static_cast<float>(v1.color.z) / 255);
			float3 color2 = make_float3(static_cast<float>(rgb[i].x) / 255,
										static_cast<float>(rgb[i].y) / 255,
										static_cast<float>(rgb[i].z) / 255);
			if (sdf * sdf > residuals_threshold) {
				if (create_mask) mask[i] = true;
				continue;
			}
			mat1x3 gradient, gradient_color;
			// x
			float voxel_size = volume->GetOptions().voxel_size;
			v1 = volume->GetInterpolatedVoxel(point +
											  make_float3(voxel_size, 0.0f, 0.0f));
			if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			tsdfvh::Voxel v2 = volume->GetInterpolatedVoxel(
					point + make_float3(-voxel_size, 0.0f, 0.0f));
			if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			gradient(0) = (v1.sdf - v2.sdf) / (2 * voxel_size);
			gradient_color(0) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);
			// y
			v1 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, voxel_size, 0.0f));
			if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			v2 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, -voxel_size, 0.0f));
			if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			gradient(1) = (v1.sdf - v2.sdf) / (2 * voxel_size);
			gradient_color(1) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);
			// z
			v1 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, 0.0f, voxel_size));
			if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			v2 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, 0.0f, -voxel_size));
			if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			gradient(2) = (v1.sdf - v2.sdf) / (2 * voxel_size);
			gradient_color(2) = ColorDifference(v1.color, v2.color) / (2 * voxel_size);

			// Partial derivative of position wrt optimization parameters
			mat3x6 d_position;
			d_position(0, 0) = 0;
			d_position(0, 1) = point.z;
			d_position(0, 2) = -point.y;
			d_position(0, 3) = 1;
			d_position(0, 4) = 0;
			d_position(0, 5) = 0;
			d_position(1, 0) = -point.z;
			d_position(1, 1) = 0;
			d_position(1, 2) = point.x;
			d_position(1, 3) = 0;
			d_position(1, 4) = 1;
			d_position(1, 5) = 0;
			d_position(2, 0) = point.y;
			d_position(2, 1) = -point.x;
			d_position(2, 2) = 0;
			d_position(2, 3) = 0;
			d_position(2, 4) = 0;
			d_position(2, 5) = 1;

			// Jacobian
			mat1x6 jacobian = gradient * d_position;
			mat1x6 jacobian_color = gradient_color * d_position;

			float huber = fabs(sdf) < huber_constant ? 1.0 : huber_constant / fabs(sdf);
			bool use_depth = true;
			bool use_color = true;
			float weight = 0.025;
			if (use_depth) {
				new_H = new_H + huber * jacobian.getTranspose() * jacobian;
				new_b = new_b + huber * jacobian.getTranspose() * sdf;
			}

			if (use_color) {
				new_H = new_H + weight * jacobian_color.getTranspose() * jacobian_color;
				new_b = new_b +
						weight * jacobian_color.getTranspose() *
						(Intensity(color) - Intensity(color2));
			}

			for (int j = 0; j < 36; j++) atomicAdd(&((*acc_H)(j)), new_H(j));
			for (int j = 0; j < 6; j++) atomicAdd(&((*acc_b)(j)), new_b(j));
		}
	}

	/**
	 * Used in camrera tracking for StaticTracker and OpticalFlowTracker. The duplication of code between this and the
	 * previous function is a very hacky way of getting the behaviour needed, but there's too much linear algebra to
	 * tidy this up right now.
	 * TODO: clean this up.
	 *
	 * @param volume
	 * @param huber_constant
	 * @param rgb
	 * @param depth
	 * @param mask
	 * @param transform
	 * @param sensor
	 * @param acc_H
	 * @param acc_b
	 * @param downsample
	 */
	__global__ void CreateLinearSystemWithMask(tsdfvh::TsdfVolume *volume,
									   float huber_constant, uchar3 *rgb,
									   float *depth, bool *mask, float4x4 transform,
									   RgbdSensor sensor, mat6x6 *acc_H,
									   mat6x1 *acc_b, int downsample)

	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int size = sensor.rows * sensor.cols;
		for (int idx = index; idx < size / (downsample * downsample); idx += stride) {
			mat6x6 new_H;
			mat6x1 new_b;
			new_H.setZero();
			new_b.setZero();
			int v = (idx / (sensor.cols / downsample)) * downsample;
			int u = (idx - (sensor.cols / downsample) * v / downsample) * downsample;
			int i = v * sensor.cols + u;
			if (depth[i] < volume->GetOptions().min_sensor_depth) {
				continue;
			}
			if (depth[i] > volume->GetOptions().max_sensor_depth) {
				continue;
			}
			float3 point = transform * GetPoint3d(i, depth[i], sensor);
			tsdfvh::Voxel v1 = volume->GetInterpolatedVoxel(point);
			if (v1.weight == 0) {
				continue;
			}
			float sdf = v1.sdf;

			// This line is the main change from the other kernel: that function was using residual thresholding here,
			// but here we're using a mask instead.
			if(mask[i])
			{
				continue;
			}
			mat1x3 gradient;
			// x
			float voxel_size = volume->GetOptions().voxel_size;
			v1 = volume->GetInterpolatedVoxel(point +
											  make_float3(voxel_size, 0.0f, 0.0f));
			if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			tsdfvh::Voxel v2 = volume->GetInterpolatedVoxel(
					point + make_float3(-voxel_size, 0.0f, 0.0f));
			if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			gradient(0) = (v1.sdf - v2.sdf) / (2 * voxel_size);
			// y
			v1 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, voxel_size, 0.0f));
			if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			v2 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, -voxel_size, 0.0f));
			if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			gradient(1) = (v1.sdf - v2.sdf) / (2 * voxel_size);
			// z
			v1 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, 0.0f, voxel_size));
			if (v1.weight == 0 || v1.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			v2 = volume->GetInterpolatedVoxel(point +
											  make_float3(0.0f, 0.0f, -voxel_size));
			if (v2.weight == 0 || v2.sdf >= volume->GetOptions().truncation_distance) {
				continue;
			}
			gradient(2) = (v1.sdf - v2.sdf) / (2 * voxel_size);

			// Partial derivative of position wrt optimization parameters
			mat3x6 d_position;
			d_position(0, 0) = 0;
			d_position(0, 1) = point.z;
			d_position(0, 2) = -point.y;
			d_position(0, 3) = 1;
			d_position(0, 4) = 0;
			d_position(0, 5) = 0;
			d_position(1, 0) = -point.z;
			d_position(1, 1) = 0;
			d_position(1, 2) = point.x;
			d_position(1, 3) = 0;
			d_position(1, 4) = 1;
			d_position(1, 5) = 0;
			d_position(2, 0) = point.y;
			d_position(2, 1) = -point.x;
			d_position(2, 2) = 0;
			d_position(2, 3) = 0;
			d_position(2, 4) = 0;
			d_position(2, 5) = 1;

			// Jacobian
			mat1x6 jacobian = gradient * d_position;

			float huber = fabs(sdf) < huber_constant ? 1.0 : huber_constant / fabs(sdf);
			bool use_depth = true;
			if (use_depth) {
				new_H = new_H + huber * jacobian.getTranspose() * jacobian;
				new_b = new_b + huber * jacobian.getTranspose() * sdf;
			}

			for (int j = 0; j < 36; j++) atomicAdd(&((*acc_H)(j)), new_H(j));
			for (int j = 0; j < 6; j++) atomicAdd(&((*acc_b)(j)), new_b(j));
		}
	}

	/**
	 * @brief Applies flooding to the mask.
	 *
	 * @param depth
	 * @param mask
	 * @param threshold
	 */
	void ApplyMaskFlood(const cv::Mat &depth, cv::Mat &mask, float threshold)
	{
		int erosion_size = 15;
		cv::Mat erosion_kernel = cv::getStructuringElement(
				cv::MORPH_ELLIPSE, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
				cv::Point(erosion_size, erosion_size));
		cv::Mat eroded_mask;
		cv::erode(mask, eroded_mask, erosion_kernel);
		std::vector <std::pair<int, int>> mask_vector;
		for (int i = 0; i < depth.rows; i++) {
			for (int j = 0; j < depth.cols; j++) {
				mask.at<uchar>(i, j) = 0;
				if (eroded_mask.at<uchar>(i, j) > 0) {
					mask_vector.push_back(std::make_pair(i, j));
				}
			}
		}

		while (!mask_vector.empty()) {
			int i = mask_vector.back().first;
			int j = mask_vector.back().second;
			mask_vector.pop_back();
			if (depth.at<float>(i, j) > 0 && mask.at<uchar>(i, j) == 0) {
				float old_depth = depth.at<float>(i, j);
				mask.at<uchar>(i, j) = 255;
				if (i - 1 >= 0) {  // up
					if (depth.at<float>(i - 1, j) > 0 && mask.at<uchar>(i - 1, j) == 0 &&
						fabs(depth.at<float>(i - 1, j) - old_depth) <
						threshold * old_depth) {
						mask_vector.push_back(std::make_pair(i - 1, j));
					}
				}
				if (i + 1 < depth.rows) {  // down
					if (depth.at<float>(i + 1, j) > 0 && mask.at<uchar>(i + 1, j) == 0 &&
						fabs(depth.at<float>(i + 1, j) - old_depth) <
						threshold * old_depth) {
						mask_vector.push_back(std::make_pair(i + 1, j));
					}
				}
				if (j - 1 >= 0) {  // left
					if (depth.at<float>(i, j - 1) > 0 && mask.at<uchar>(i, j - 1) == 0 &&
						fabs(depth.at<float>(i, j - 1) - old_depth) <
						threshold * old_depth) {
						mask_vector.push_back(std::make_pair(i, j - 1));
					}
				}
				if (j + 1 < depth.cols) {  // right
					if (depth.at<float>(i, j + 1) > 0 && mask.at<uchar>(i, j + 1) == 0 &&
						fabs(depth.at<float>(i, j + 1) - old_depth) <
						threshold * old_depth) {
						mask_vector.push_back(std::make_pair(i, j + 1));
					}
				}
			}
		}
	}

	/**
	 * @brief Returns a HSV image of the optical flow field.
	 *
	 * @param flow
	 * @return
	 */
	cv::Mat visualizeFLowField(cv::Mat flow)
	{
		cv::Mat flow_parts[2];
		split(flow, flow_parts);
		cv::Mat magnitude, angle, magn_norm;
		cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
		normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
		angle *= ((1.f / 360.f) * (180.f / 255.f));

		//build hsv image
		cv::Mat _hsv[3], hsv, hsv8, bgr;
		_hsv[0] = angle;
		_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magn_norm;
		merge(_hsv, 3, hsv);
		hsv.convertTo(hsv8, CV_8U, 255.0);

		cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

		return bgr;
	}

	__host__ __device__ Eigen::Vector4d projectIntoWorldSpace(int x, int y, float depth, RgbdSensor &sensor)
	{
		// Project the 2d point into 3d space
		Eigen::Vector4d point;
		point(0) = (x - sensor.cx) * depth / sensor.fx;
		point(1) = (y - sensor.cy) * depth / sensor.fy;
		point(2) = depth;
		point(3) = 1.0f;

		return point;
	}

	__host__ __device__ Eigen::Vector2d projectIntoImageSpace(Eigen::Vector4d point, RgbdSensor &sensor)
	{
		// Project the 3d point into 2d space
		Eigen::Vector2d point2d;
		point2d(0) = (sensor.fx * point(0) / point(2)) + sensor.cx;
		point2d(1) = (sensor.fy * point(1) / point(2)) + sensor.cy;

		return point2d;
	}

	/**
	 * @brief Get the current pose of the camera.
	 *
	 * @return the current pose of the camera.
	 */
	Eigen::Matrix4d Tracker::GetCurrentPose()
	{
		return pose_;
	}

	/**
	 * @brief Extract a mesh from the current volume.
	 *
	 * @param lower_corner
	 * @param upper_corner
	 * @return the mesh.
	 */
	tsdfvh::Mesh Tracker::ExtractMesh(const float3 &lower_corner,
									  const float3 &upper_corner)
	{
		return volume_->ExtractMesh(lower_corner, upper_corner);
	}

	/**
	 * @brief Generate a virtual RGB image from the current pose.
	 *
	 * @param width
	 * @param height
	 * @return the virtual RGB image.
	 */
	cv::Mat Tracker::GenerateRgb(int width, int height)
	{
		Eigen::Matrix4f posef = pose_.cast<float>();
		float4x4 pose_cuda = float4x4(posef.data()).getTranspose();
		RgbdSensor virtual_sensor;
		virtual_sensor.rows = height;
		virtual_sensor.cols = width;
		virtual_sensor.depth_factor = sensor_.depth_factor;
		float factor_x = static_cast<float>(virtual_sensor.cols) /
						 static_cast<float>(sensor_.cols);
		float factor_y = static_cast<float>(virtual_sensor.rows) /
						 static_cast<float>(sensor_.rows);
		virtual_sensor.fx = factor_x * sensor_.fx;
		virtual_sensor.fy = factor_y * sensor_.fy;
		virtual_sensor.cx = factor_x * sensor_.cx;
		virtual_sensor.cy = factor_y * sensor_.cy;
		uchar3 *virtual_rgb = volume_->GenerateRgb(pose_cuda, virtual_sensor);

		cv::Mat cv_virtual_rgb(virtual_sensor.rows, virtual_sensor.cols, CV_8UC3);
		for (int i = 0; i < virtual_sensor.rows; i++) {
			for (int j = 0; j < virtual_sensor.cols; j++) {
				cv_virtual_rgb.at<cv::Vec3b>(i, j)[2] =
						virtual_rgb[i * virtual_sensor.cols + j].x;
				cv_virtual_rgb.at<cv::Vec3b>(i, j)[1] =
						virtual_rgb[i * virtual_sensor.cols + j].y;
				cv_virtual_rgb.at<cv::Vec3b>(i, j)[0] =
						virtual_rgb[i * virtual_sensor.cols + j].z;
			}
		}

		return cv_virtual_rgb;
	}

	/**
	 * Logs the mask to an output video if the options are set.
	 *
	 * @param mask
	 * @param image
	 */
	void Tracker::LogMask(bool *mask, RgbdImage &image)
	{
		if (options_.output_mask_video) {
			cv::Mat output_mask(image.sensor_.rows, image.sensor_.cols, CV_8UC1);

			for (int i = 0; i < image.sensor_.rows; i++) {
				for (int j = 0; j < image.sensor_.cols; j++) {
					if (mask[i * image.sensor_.cols + j]) {
						output_mask.at<uchar>(i, j) = 255;
					}
					else {
						output_mask.at<uchar>(i, j) = 0;
					}
				}
			}

			cv::cvtColor(output_mask, output_mask, CV_GRAY2BGR);
			logger_->addFrameToOutputVideo(output_mask, "mask_output.avi");
		}
	}

	/**
	 * Logs the flow to an output file if the options are set.
	 *
	 * @param flowField
	 * @param name
	 */
	void Tracker::LogFlowField(cv::Mat &flowField, string name)
	{
		if (options_.output_flow_video) {
			cv::Mat flowFieldFrame = visualizeFLowField(flowField);
			logger_->addFrameToOutputVideo(flowFieldFrame, name);
		}
	}

	/**
	 * @brief Constructor for ReTracker.
	 *
	 * @param tsdf_options
	 * @param tracker_options
	 * @param sensor
	 * @param logger
	 */
	ReTracker::ReTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
					 const TrackerOptions &tracker_options,
					 const RgbdSensor &sensor,
					 Logger *logger) : Tracker(tsdf_options, tracker_options, sensor, logger)
	{}

	/**
	 * @brief Track the camera using the given image.
	 *
	 * @param image
	 */
	void ReTracker::TrackCamera(const RgbdImage &image, bool *mask, bool create_mask)
	{
		Vector6d increment, prev_increment;
		increment << 0, 0, 0, 0, 0, 0;
		prev_increment = increment;

		mat6x6 *acc_H;
		cudaMallocManaged(&acc_H, sizeof(mat6x6));
		mat6x1 *acc_b;
		cudaMallocManaged(&acc_b, sizeof(mat6x1));
		cudaDeviceSynchronize();
		for (int lvl = 0; lvl < 3; ++lvl) {
			for (int i = 0; i < options_.max_iterations_per_level[lvl]; ++i) {
				Eigen::Matrix4d cam_to_world = Exp(v2t(increment)) * pose_;
				Eigen::Matrix4f cam_to_worldf = cam_to_world.cast<float>();
				float4x4 transform_cuda = float4x4(cam_to_worldf.data()).getTranspose();

				acc_H->setZero();
				acc_b->setZero();
				int threads_per_block = THREADS_PER_BLOCK3;
				int thread_blocks =
						(sensor_.cols * sensor_.rows + threads_per_block - 1) /
						threads_per_block;
				bool create_mask_now =
						(lvl == 2) && (i == (options_.max_iterations_per_level[2] - 1)) &&
						create_mask;

				float residuals_threshold = 0;
				residuals_threshold = volume_->GetOptions().truncation_distance *
									  volume_->GetOptions().truncation_distance / 2;
				if (!create_mask) {
					residuals_threshold = volume_->GetOptions().truncation_distance *
										  volume_->GetOptions().truncation_distance;
				}
				// Kernel to fill in parallel acc_H and acc_b
				CreateLinearSystemResidual<<<thread_blocks, threads_per_block>>>(
						volume_, options_.huber_constant, image.rgb_, image.depth_, mask,
						transform_cuda, sensor_, acc_H, acc_b, options_.downsample[lvl],
						residuals_threshold, create_mask_now);
				cudaDeviceSynchronize();
				Eigen::Matrix<double, 6, 6> H;
				Vector6d b;
				for (int r = 0; r < 6; r++) {
					for (int c = 0; c < 6; c++) {
						H(r, c) = static_cast<double>((*acc_H)(r, c));
					}
				}
				for (int k = 0; k < 6; k++) {
					b(k) = static_cast<double>((*acc_b)(k));
				}
				double scaling = 1 / H.maxCoeff();
				b *= scaling;
				H *= scaling;
				H = H + options_.regularization * Eigen::MatrixXd::Identity(6, 6) * i;
				increment = increment - SolveLdlt(H, b);
				Vector6d change = increment - prev_increment;
				if (change.norm() <= options_.min_increment) break;
				prev_increment = increment;
			}
		}
		if (std::isnan(increment.sum())) increment << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

		cudaFree(acc_H);
		cudaFree(acc_b);

		pose_ = Exp(v2t(increment)) * pose_;
		prev_increment_ = increment;
	}

	/**
	 * @brief Add a new scan to the tracker.
	 *
	 * @param rgb
	 * @param depth
	 */
	void ReTracker::AddScan(const cv::Mat &rgb, const cv::Mat &depth)
	{
		RgbdImage image;
		image.Init(sensor_);

		// Linear copy for now
		for (int i = 0; i < image.sensor_.rows; i++) {
			for (int j = 0; j < image.sensor_.cols; j++) {
				image.rgb_[i * image.sensor_.cols + j] = make_uchar3(rgb.at<cv::Vec3b>(i, j)(2), rgb.at<cv::Vec3b>(i, j)(1), rgb.at<cv::Vec3b>(i, j)(0));
				image.depth_[i * image.sensor_.cols + j] = depth.at<float>(i, j);
			}
		}

		bool *mask;
		cudaMallocManaged(&mask, sizeof(bool) * image.sensor_.rows * image.sensor_.cols);
		for (int i = 0; i < image.sensor_.rows * image.sensor_.cols; i++) {
			mask[i] = false;
		}

		if (!first_scan_) {
			Eigen::Matrix4d prev_pose = pose_;
			TrackCamera(image, mask, true);

			cv::Mat cvmask(image.sensor_.rows, image.sensor_.cols, CV_8UC1);
			for (int i = 0; i < image.sensor_.rows; i++) {
				for (int j = 0; j < image.sensor_.cols; j++) {
					if (mask[i * image.sensor_.cols + j]) {
						cvmask.at<uchar>(i, j) = 255;
					} else {
						cvmask.at<uchar>(i, j) = 0;
					}
				}
			}

			ApplyMaskFlood(depth, cvmask, 0.007);

			int dilation_size = 10;
			cv::Mat dilation_kernel = cv::getStructuringElement(
				cv::MORPH_ELLIPSE, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
				cv::Point(dilation_size, dilation_size));
			cv::dilate(cvmask, cvmask, dilation_kernel);

			for (int i = 0; i < image.sensor_.rows; i++) {
				for (int j = 0; j < image.sensor_.cols; j++) {
					if (cvmask.at<uchar>(i, j) > 0) {
						mask[i * image.sensor_.cols + j] = true;
					} else {
						mask[i * image.sensor_.cols + j] = false;
					}
				}
			}

			pose_ = prev_pose;
			TrackCamera(image, mask, false);
		} else {
			first_scan_ = false;
		}

		Eigen::Matrix4f posef = pose_.cast<float>();
		float4x4 pose_cuda = float4x4(posef.data()).getTranspose();

		LogMask(mask, image);

		volume_->IntegrateScan(image, pose_cuda, mask);

		cudaFree(mask);
	}


	/**
	 * @brief Constructor for StaticTracker.
	 *
	 * @param tsdf_options
	 * @param tracker_options
	 * @param sensor
	 * @param logger
	 */
	StaticTracker::StaticTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
					 const TrackerOptions &tracker_options,
					 const RgbdSensor &sensor,
					 Logger *logger) : Tracker(tsdf_options, tracker_options, sensor, logger)
	{}

	/**
	 * @brief Track the camera to the current scan.
	 *
	 * @param image
	 */
	void StaticTracker::TrackCamera(const RgbdImage &image, bool *mask)
	{
		Vector6d increment, prev_increment;
		increment << 0, 0, 0, 0, 0, 0;
		prev_increment = increment;

		mat6x6 *acc_H;
		cudaMallocManaged(&acc_H, sizeof(mat6x6));
		mat6x1 *acc_b;
		cudaMallocManaged(&acc_b, sizeof(mat6x1));
		cudaDeviceSynchronize();
		for (int lvl = 0; lvl < 3; ++lvl) {
			for (int i = 0; i < options_.max_iterations_per_level[lvl]; ++i) {
				Eigen::Matrix4d cam_to_world = Exp(v2t(increment)) * pose_;
				Eigen::Matrix4f cam_to_worldf = cam_to_world.cast<float>();
				float4x4 transform_cuda = float4x4(cam_to_worldf.data()).getTranspose();

				acc_H->setZero();
				acc_b->setZero();
				int threads_per_block = THREADS_PER_BLOCK3;
				int thread_blocks =
						(sensor_.cols * sensor_.rows + threads_per_block - 1) /
						threads_per_block;

				// Kernel to fill in parallel acc_H and acc_b
				CreateLinearSystemWithMask<<<thread_blocks, threads_per_block>>>(
						volume_, options_.huber_constant, image.rgb_, image.depth_, mask,
						transform_cuda, sensor_, acc_H, acc_b, options_.downsample[lvl]);
				cudaDeviceSynchronize();
				Eigen::Matrix<double, 6, 6> H;
				Vector6d b;
				for (int r = 0; r < 6; r++) {
					for (int c = 0; c < 6; c++) {
						H(r, c) = static_cast<double>((*acc_H)(r, c));
					}
				}
				for (int k = 0; k < 6; k++) {
					b(k) = static_cast<double>((*acc_b)(k));
				}
				double scaling = 1 / H.maxCoeff();
				b *= scaling;
				H *= scaling;
				H = H + options_.regularization * Eigen::MatrixXd::Identity(6, 6) * i;
				increment = increment - SolveLdlt(H, b);
				Vector6d change = increment - prev_increment;
				if (change.norm() <= options_.min_increment) break;
				prev_increment = increment;
			}
		}
		if (std::isnan(increment.sum())) increment << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

		cudaFree(acc_H);
		cudaFree(acc_b);

		pose_ = Exp(v2t(increment)) * pose_;
		prev_increment_ = increment;
	}

	/**
	 * @brief Adds the given image to the volume.
	 *
	 * @param image
	 * @param mask
	 * @param use_prev_increment
	 */
	void StaticTracker::AddScan(const cv::Mat &rgb, const cv::Mat &depth)
	{
		RgbdImage image;
		image.Init(sensor_);

		// Linear copy for now
		for (int i = 0; i < image.sensor_.rows; i++) {
			for (int j = 0; j < image.sensor_.cols; j++) {
				image.rgb_[i * image.sensor_.cols + j] = make_uchar3(rgb.at<cv::Vec3b>(i, j)(2), rgb.at<cv::Vec3b>(i, j)(1), rgb.at<cv::Vec3b>(i, j)(0));
				image.depth_[i * image.sensor_.cols + j] = depth.at<float>(i, j);
			}
		}

		bool *mask;
		cudaMallocManaged(&mask, sizeof(bool) * image.sensor_.rows * image.sensor_.cols);
		for (int i = 0; i < image.sensor_.rows * image.sensor_.cols; i++) {
			mask[i] = false;
		}

		if (!first_scan_) {
			Eigen::Matrix4d prev_pose = pose_;
			TrackCamera(image, mask);
		} else {
			first_scan_ = false;
		}

		Eigen::Matrix4f posef = pose_.cast<float>();
		float4x4 pose_cuda = float4x4(posef.data()).getTranspose();

		LogMask(mask, image);

		volume_->IntegrateScan(image, pose_cuda, mask);

		cudaFree(mask);
	}


	/**
	 * @brief Constructor for the OpticalFlowTracker class.
	 *
	 * @param image
	 */
	OpticalFlowTracker::OpticalFlowTracker(const tsdfvh::TsdfVolumeOptions &tsdf_options,
					 const TrackerOptions &tracker_options,
					 const RgbdSensor &sensor,
					 Logger *logger) : Tracker(tsdf_options, tracker_options, sensor, logger)
	{}

	/**
	 * @brief Runs on the GPU to calculate optical flow between the previous (P) and current (C) images.
	 * @param P
	 * @param C
	 * @return
	 */
	void GPUOpticalFlow(RgbdImage &P, RgbdImage &C, float *flow)
	{
		cv::cuda::GpuMat current, previous, flowMat;
		cv::cuda::GpuMat d_P{P.sensor_.rows, P.sensor_.cols, CV_8UC3, P.rgb_};
		cv::cuda::GpuMat d_C{C.sensor_.rows, C.sensor_.cols, CV_8UC3, C.rgb_};

		cv::cuda::cvtColor(d_C, current, cv::COLOR_BGR2GRAY);
		cv::cuda::cvtColor(d_P, previous, cv::COLOR_BGR2GRAY);

		cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
		farn->setPyrScale(0.5);
		farn->setNumLevels(3);
		farn->setWinSize(15);
		farn->setNumIters(3);
		farn->setPolyN(5);
		farn->setPolySigma(1.2);

		farn->calc(previous, current, flowMat);

		// TODO: This feels like a really dumb thing to have to download back to the host before copying back onto device
		// See if there is a way to improve this.
		cv::Mat flow_cpu;
		flowMat.download(flow_cpu);

		for(int flowIndex = 0; flowIndex < P.sensor_.rows * P.sensor_.cols; flowIndex++) {
			flow[flowIndex * 2] = flow_cpu.at<cv::Point2f>(flowIndex).x;
			flow[flowIndex * 2 + 1] = flow_cpu.at<cv::Point2f>(flowIndex).y;
		}
	}

	/**
	 * Subtracts the egomotion from the optical flow to get the difference flow and stores this in the difference_flow array.
	 *
	 * @param increment
	 * @param prev_image
	 * @param optical_flow
	 * @param difference_flow
	 */
	__global__ void SubtractEgomotion(Eigen::Matrix4d increment, RgbdImage prev_image, float *optical_flow, float *difference_flow)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int size = prev_image.sensor_.rows * prev_image.sensor_.cols;

		for (int idx = index; idx < size; idx += stride) {
			difference_flow[idx * 2] = 0;
			difference_flow[idx * 2 + 1] = 0;

			if(sqrt((optical_flow[idx * 2] * optical_flow[idx * 2]) + (optical_flow[idx * 2 + 1] * optical_flow[idx * 2 + 1])) < OPTICAL_FLOW_MOVEMENT_THRESHOLD) {
				difference_flow[idx * 2] = optical_flow[idx * 2];
				difference_flow[idx * 2 + 1] = optical_flow[idx * 2 + 1];
				continue;
			}
			if (prev_image.depth_[idx] == 0) {
				difference_flow[idx * 2] = 0;
				difference_flow[idx * 2 + 1] = 0;
				continue;
			}

			int x = idx % prev_image.sensor_.cols;
			int y = idx / prev_image.sensor_.cols;

			Eigen::Vector4d point = projectIntoWorldSpace(x, y, prev_image.depth_[idx], prev_image.sensor_);
			Eigen::Vector4d transformedPoint = increment * point;
			transformedPoint = transformedPoint / transformedPoint(3);
			Eigen::Vector2d transformedPoint2D = projectIntoImageSpace(transformedPoint, prev_image.sensor_);

//			float X = (float)x - transformedPoint2D.x();
//			float Y = (float)y - transformedPoint2D.y();

			float X = transformedPoint2D.x() - (float)x;
			float Y = transformedPoint2D.y() - (float)y;

			difference_flow[idx * 2] = optical_flow[idx * 2] - X;
			difference_flow[idx * 2 + 1] = optical_flow[idx * 2 + 1] - Y;
		}
	}

	/**
	 * Thresholds the given difference flow to determine which pixels are moving.
	 *
	 * @param difference_flow
	 * @param mask
	 * @param image
	 */
	__global__ void ThresholdOpticalFlow(float *difference_flow, bool *mask, RgbdImage image)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int size = image.sensor_.rows * image.sensor_.cols;
		for (int idx = index; idx < size; idx += stride) {
			if(sqrt((difference_flow[idx * 2] * difference_flow[idx * 2]) + (difference_flow[idx * 2 + 1] * difference_flow[idx * 2 + 1])) > OPTICAL_FLOW_MOVEMENT_THRESHOLD * image.depth_[idx]) {
				mask[idx] = true;
			}
			else
			{
				mask[idx] = false;
			}
		}
	}

	/**
	 * @brief Tracks the camera using optical flow.
	 *
	 * @param image
	 * @param mask
	 */
	void OpticalFlowTracker::TrackCamera(const RgbdImage &image, bool *mask)
	{
		Vector6d increment, prev_increment;
		increment << 0, 0, 0, 0, 0, 0;
		prev_increment = increment;

		mat6x6 *acc_H;
		cudaMallocManaged(&acc_H, sizeof(mat6x6));
		mat6x1 *acc_b;
		cudaMallocManaged(&acc_b, sizeof(mat6x1));
		cudaDeviceSynchronize();
		for (int lvl = 0; lvl < 3; ++lvl) {
			for (int i = 0; i < options_.max_iterations_per_level[lvl]; ++i) {
				Eigen::Matrix4d cam_to_world = Exp(v2t(increment)) * pose_;
				Eigen::Matrix4f cam_to_worldf = cam_to_world.cast<float>();
				float4x4 transform_cuda = float4x4(cam_to_worldf.data()).getTranspose();

				acc_H->setZero();
				acc_b->setZero();
				int threads_per_block = THREADS_PER_BLOCK3;
				int thread_blocks =
						(sensor_.cols * sensor_.rows + threads_per_block - 1) /
						threads_per_block;

				// Kernel to fill in parallel acc_H and acc_b
				CreateLinearSystemWithMask<<<thread_blocks, threads_per_block>>>(
						volume_, options_.huber_constant, image.rgb_, image.depth_, mask,
						transform_cuda, sensor_, acc_H, acc_b, options_.downsample[lvl]);
				cudaDeviceSynchronize();
				Eigen::Matrix<double, 6, 6> H;
				Vector6d b;
				for (int r = 0; r < 6; r++) {
					for (int c = 0; c < 6; c++) {
						H(r, c) = static_cast<double>((*acc_H)(r, c));
					}
				}
				for (int k = 0; k < 6; k++) {
					b(k) = static_cast<double>((*acc_b)(k));
				}
				double scaling = 1 / H.maxCoeff();
				b *= scaling;
				H *= scaling;
				H = H + options_.regularization * Eigen::MatrixXd::Identity(6, 6) * i;
				increment = increment - SolveLdlt(H, b);
				Vector6d change = increment - prev_increment;
				if (change.norm() <= options_.min_increment) break;
				prev_increment = increment;
			}
		}
		if (std::isnan(increment.sum())) increment << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

		cudaFree(acc_H);
		cudaFree(acc_b);

		pose_ = Exp(v2t(increment)) * pose_;
		prev_increment_ = increment;
	}

	/**
	 * @brief Calculates the mask using optical flow.
	 *
	 * @param image
	 * @param mask
	 */
	void OpticalFlowTracker::calculateMask(RgbdImage image, bool *mask)
	{
		// Allocate GPU memory
		float *d_optical_flow;
		cudaMallocManaged(&d_optical_flow, sizeof(float) * 2 * image.sensor_.rows * image.sensor_.cols);

		float *d_optical_flow_sans_egomotion;
		cudaMallocManaged(&d_optical_flow_sans_egomotion, sizeof(float) * 2 * image.sensor_.rows * image.sensor_.cols);

		// Kernel settings
		int threads_per_block = THREADS_PER_BLOCK3;
		int thread_blocks = (image.sensor_.cols * image.sensor_.rows + threads_per_block - 1) / threads_per_block;

		// Calculate the optical flow
		GPUOpticalFlow(prev_image, image, d_optical_flow);
		cudaDeviceSynchronize();
		// Store the previous pose before we update the pose.
		Eigen::Matrix4d previousPose = pose_;
		Eigen::Matrix4d previousPoseInverse = previousPose.inverse();

		pose_ = previousPose;
		TrackCamera(image, mask);

		// Subtract the egomotion from the optical flow
		Eigen::Matrix4d increment = pose_ * previousPoseInverse;

		SubtractEgomotion<<<thread_blocks, threads_per_block>>>(increment, prev_image, d_optical_flow, d_optical_flow_sans_egomotion);
		cudaDeviceSynchronize();

		// Now we can seed the mask by thresholding the output of the subtracted egomotion.
		ThresholdOpticalFlow<<<thread_blocks, threads_per_block>>>(d_optical_flow_sans_egomotion, mask, image);
		cudaDeviceSynchronize();

		// Perform morphological closing on the mask:
		cv::Mat cvmask = cv::Mat(image.sensor_.rows, image.sensor_.cols, CV_8UC1);
		cv::Mat depth = cv::Mat(image.sensor_.rows, image.sensor_.cols, CV_32FC1);

		for (int i = 0; i < image.sensor_.rows; i++) {
			for (int j = 0; j < image.sensor_.cols; j++) {
				cvmask.at<uchar>(i, j) = mask[i * image.sensor_.cols + j] ? 255 : 0;
				depth.at<float>(i, j) = image.depth_[i * image.sensor_.cols + j];
			}
		}

		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(17, 17));
		cv::erode(cvmask, cvmask, element);
		cv::dilate(cvmask, cvmask, element);

		// Finally we must perform the flood fill to ensure that the mask is connected.
		queue<tuple<int, int, int>> q;
		for (int i = 0; i < image.sensor_.rows; i++) {
			for (int j = 0; j < image.sensor_.cols; j++) {
				if (cvmask.at<uchar>(i, j) == 255) {
					q.push(make_tuple(i, j, 1));
				}
			}
		}

		float depthThreshold = 0.2f;
		int growthThreshold = 75;

		while(!q.empty()) {
			tuple<int, int, int> t = q.front();
			q.pop();
			int i = get<0>(t);
			int j = get<1>(t);
			int growth = get<2>(t);

			if (growth > growthThreshold) continue;

			int depthIndex = i * image.sensor_.cols + j;

			if (i > 0 && cvmask.at<uchar>(i - 1, j) == 0 && abs(depth.at<float>(i - 1, j) - depth.at<float>(i, j)) < depthThreshold) {
				cvmask.at<uchar>(i - 1, j) = 255;
				q.push(make_tuple(i - 1, j, growth + 1));
			}

			if (i < image.sensor_.rows - 1 && cvmask.at<uchar>(i + 1, j) == 0 && abs(depth.at<float>(i + 1, j) - depth.at<float>(i, j)) < depthThreshold) {
				cvmask.at<uchar>(i + 1, j) = 255;
				q.push(make_tuple(i + 1, j, growth + 1));
			}

			if (j > 0 && cvmask.at<uchar>(i, j - 1) == 0 && abs(depth.at<float>(i, j - 1) - depth.at<float>(i, j)) < depthThreshold) {
				cvmask.at<uchar>(i, j - 1) = 255;
				q.push(make_tuple(i, j - 1, growth + 1));
			}

			if (j < image.sensor_.cols - 1 && cvmask.at<uchar>(i, j + 1) == 0 && abs(depth.at<float>(i, j + 1) - depth.at<float>(i, j)) < depthThreshold) {
				cvmask.at<uchar>(i, j + 1) = 255;
				q.push(make_tuple(i, j + 1, growth + 1));
			}
		}

		for (int i = 0; i < image.sensor_.rows; i++) {
			for (int j = 0; j < image.sensor_.cols; j++) {
				mask[i * image.sensor_.cols + j] = cvmask.at<uchar>(i, j) == 255;
			}
		}

		// Free the memory allocated on GPU
		cudaFree(d_optical_flow);
		cudaFree(d_optical_flow_sans_egomotion);
	}

	/**
	 * @brief Add a new scan to the tracker. This will perform the optical flow tracking and update the pose.
	 *
	 * @param rgb
	 * @param depth
	 */
	void OpticalFlowTracker::AddScan(const cv::Mat &rgb, const cv::Mat &depth)
	{
		RgbdImage image;
		image.Init(sensor_, true); // Passing the true flag here to perform manual memory management

		cudaMallocManaged(&image.rgb_, sizeof(uchar3) * image.sensor_.rows * image.sensor_.cols);
		cudaMallocManaged(&image.depth_, sizeof(float) * image.sensor_.rows * image.sensor_.cols);

		// Copy the data from the cv::Mat to the RgbdImage. This allows us to use the same datastructures for the CPU and GPU
		for(int datum = 0; datum < image.sensor_.rows * image.sensor_.cols; datum++) {
			image.rgb_[datum] = make_uchar3(rgb.at<cv::Vec3b>(datum)(2), rgb.at<cv::Vec3b>(datum)(1), rgb.at<cv::Vec3b>(datum)(0));
			image.depth_[datum] = depth.at<float>(datum);
		}

		// Declare the mask to be used for this scan.
		bool *mask;
		cudaMallocManaged(&mask, sizeof(bool) * image.sensor_.rows * image.sensor_.cols);
		cudaDeviceSynchronize();
		// If this isn't the first scan, then we can use optical flow to create a mask and track the camera. Otherwise,
		// we just integrate the scan into the volume and flip the first_scan_ flag.
		if(!first_scan_) {
			// First we calculate the mask.
			calculateMask(image, mask);
			// Now we use the mask to track the camera
			TrackCamera(image, mask);

			// As this isn't the first scan, we have a previous image (which we are about to overwrite with the
			// current image) so we need to free the memory allocated for the previous image.
			// These lines are needed as we created the memory for the previous image in manual mode.
			cudaFree(prev_image.rgb_);
			cudaFree(prev_image.depth_);
		}
		else {
			first_scan_ = false;
		}

		// Store the current as the previous image for the next call of this function.
		prev_image = image;

		// Integrate the scan into the volume using the calculated pose and mask.
		Eigen::Matrix4f posef = pose_.cast<float>();
		float4x4 pose_cuda = float4x4(posef.data()).getTranspose();
		volume_->IntegrateScan(image, pose_cuda, mask);

		// Log and free the mask
		LogMask(mask, image);
		cudaFree(mask);
	}
}
