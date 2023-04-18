// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "tracker/tracker.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include "tracker/eigen_wrapper.h"
#include "utils/matrix_utils.h"
#include "utils/utils.h"
#include "utils/rgbd_image.h"

#define THREADS_PER_BLOCK3 32

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
	cv::Mat GPUOpticalFlow(cv::Mat P, cv::Mat C)
	{
		cv::cuda::GpuMat current, previous, flow;
		cv::cuda::GpuMat d_P(P);
		cv::cuda::GpuMat d_C(C);

		cv::cuda::cvtColor(d_C, current, cv::COLOR_BGR2GRAY);
		cv::cuda::cvtColor(d_P, previous, cv::COLOR_BGR2GRAY);

		cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
		farn->setPyrScale(0.5);
		farn->setNumLevels(3);
		farn->setWinSize(15);
		farn->setNumIters(3);
		farn->setPolyN(5);
		farn->setPolySigma(1.2);

		farn->calc(previous, current, flow);

		cv::Mat result;
		flow.download(result);

		return result;
	}

	/**
	 * @brief Runs on the GPU to estimate optical flow using estimated camera pose.
	 *
	 * @param increment
	 * @param prev_depth_frame
	 * @param sensor
	 * @param flow
	 */
	__global__ void estimate_flow_kernel(Eigen::Matrix4d increment, float *prev_depth_frame, RgbdSensor sensor, float *flow)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		int size = sensor.rows * sensor.cols;
		for (int idx = index; idx < size; idx += stride) {
			if (prev_depth_frame[idx] == 0) { continue; }

			int x = idx % sensor.cols;
			int y = idx / sensor.cols;

			Eigen::Vector4d point = projectIntoWorldSpace(x, y, prev_depth_frame[idx], sensor);
			Eigen::Vector4d transformedPoint = (increment) * point;
			transformedPoint = transformedPoint / transformedPoint(3);
			Eigen::Vector2d transformedPoint2D = projectIntoImageSpace(transformedPoint, sensor);

			float X = x - transformedPoint2D.x();
			float Y = y - transformedPoint2D.y();

			flow[idx * 2] = X;
        	flow[idx * 2 + 1] = y - transformedPoint2D.y();
		}
	}

	/**
	 * @brief Creates an "optical flow" estimate of the image by applying the transformation (from CPE) to the previous
	 * image.
	 *
	 * @param increment
	 * @param prev_depth_frame
	 * @param sensor
	 * @return
	 */
	cv::Mat EstimateFlowWithCPE(Eigen::Matrix4d increment, cv::Mat prev_depth_frame, RgbdSensor sensor)
	{
		// Allocate memory on the device. We multiply by 2 because we need to store the x and y components of the flow.
		float *flow_d;
		cudaMallocManaged(&flow_d, sizeof(float) * sensor.rows * sensor.cols * 2);
		for(int point = 0; point < sensor.rows * sensor.cols; point++)
		{
			flow_d[point] = 0;
		}

		// Allocate memory on the GPU for the depth array
		float *depth_d;
		cudaMallocManaged(&depth_d, sizeof(float) * prev_depth_frame.rows * prev_depth_frame.cols);
		for(int point = 0; point < sensor.rows * sensor.cols; point++)
		{
			depth_d[point] = prev_depth_frame.at<float>(point);
		}

		int threads_per_block = THREADS_PER_BLOCK3;
		int thread_blocks = (sensor.cols * sensor.rows + threads_per_block - 1) / threads_per_block;
		estimate_flow_kernel<<<thread_blocks, threads_per_block>>>(increment, depth_d, sensor, flow_d);
		cudaDeviceSynchronize();

		// Copy the data back to the host
		cv::Mat flow(prev_depth_frame.rows, prev_depth_frame.cols, CV_32FC2, cv::Scalar(0,0));
		for(int point = 0; point < sensor.rows * sensor.cols; point++)
		{
			flow.at<cv::Point2f>(point) = cv::Point2f{flow_d[point * 2], flow_d[point * 2 + 1]};
		}

		cudaFree(flow_d);
		return flow;
	}

	//TODO: Implement on GPU
//	__global__ void DifferenceFlowFramesKernel(float *opticalFlow, float *estimatedFlow, float *difference)
//	{
//		int index = blockIdx.x * blockDim.x + threadIdx.x;
//		int stride = blockDim.x * gridDim.x;
//		int size = opticalFlow.rows * opticalFlow.cols;
//		for (int idx = index; idx < size; idx += stride) {
//			if(opticalFlow[idx * 2] == 0 && opticalFlow[idx * 2 + 1] == 0) { continue; }
//			if(estimatedFlow[idx * 2] == 0 && estimatedFlow[idx * 2 + 1] == 0) { continue; }
//
//			difference[idx * 2] = opticalFlow[idx * 2] - estimatedFlow[idx * 2];
//			difference[idx * 2 + 1] = opticalFlow[idx * 2 + 1] - estimatedFlow[idx * 2 + 1];
//		}
//	}

	cv::Mat DifferenceFlowFrames(cv::Mat opticalFlow, cv::Mat estimatedFlow)
	{
		// TODO: Implement on GPU
//
//		cv::cuda::GpuMat opticalFlowGpu{opticalFlow};
//		cv::cuda::GpuMat estimatedFlowGpu{estimatedFlow};
//		cv::cuda::GpuMat differenceFlowGpu{differenceFlow};
//
//		int threads_per_block = THREADS_PER_BLOCK3;
//		int thread_blocks = (opticalFlow.rows * opticalFlow.cols + threads_per_block - 1) / threads_per_block;
//		DifferenceFlowFramesKernel<<<thread_blocks, threads_per_block>>>(opticalFlowGpu, estimatedFlowGpu, differenceFlowGpu);
//		cudaDeviceSynchronize();
//
//		differenceFlowGpu.download(differenceFlow);
//		return differenceFlow;

		cv::Mat differenceFlow{opticalFlow.size(), opticalFlow.type()};

		// Now subtract the pose flow from the optical flow, only if the optical flow is larger than a threshold:
		for (int i = 0; i < opticalFlow.rows; i++) {
			for (int j = 0; j < opticalFlow.cols; j++) {
				// If the sensor didn't detect a depth value for a pixel, then we have no flow estimate. We pass this noise
				// on to the difference flow.
				if(length(estimatedFlow.at<cv::Point2f>(i, j)) < 1) {
					differenceFlow.at<cv::Point2f>(i, j) = cv::Point2f(0, 0);
				}
				// If the optical flow is small, then motion is already below a threshold
				else if (length(opticalFlow.at<cv::Point2f>(i, j)) < 3) {
					differenceFlow.at<cv::Point2f>(i, j) = opticalFlow.at<cv::Point2f>(i, j);
				}
				else {
					differenceFlow.at<cv::Point2f>(i, j) = opticalFlow.at<cv::Point2f>(i, j) - estimatedFlow.at<cv::Point2f>(i, j);
				}
			}
		}

		return differenceFlow;
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


	void OpticalFlowTracker::AddScan(const cv::Mat &rgb, const cv::Mat &depth)
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
			cv::Mat farnbackFlowField = GPUOpticalFlow(prev_rgb_frame, rgb);
			LogFlowField(farnbackFlowField, "optical-flow");

			Eigen::Matrix4d previousPose = pose_;
			TrackCamera(image, mask);

			Eigen::Matrix4d inverse = previousPose.inverse();
			Eigen::Matrix4d increment = pose_ * inverse;

			cv::Mat poseFlowB = EstimateFlowWithCPE(increment, prev_depth_frame, sensor_);
			LogFlowField(poseFlowB, "pose-flow");

			cv::Mat differenceFlow = DifferenceFlowFrames(farnbackFlowField, poseFlowB);
			LogFlowField(differenceFlow, "difference-flow");

			for (int i = 0; i < image.sensor_.rows; i++) {
				for (int j = 0; j < image.sensor_.cols; j++) {
					if (length(differenceFlow.at<cv::Point2f>(i, j)) > (3 / depth.at<float>(i, j))) {
						mask[i * image.sensor_.cols + j] = true;
					}
				}
			}
		} else {
			first_scan_ = false;
		}

		prev_depth_frame = depth;
		prev_rgb_frame = rgb;

		Eigen::Matrix4f posef = pose_.cast<float>();
		float4x4 pose_cuda = float4x4(posef.data()).getTranspose();

		LogMask(mask, image);

		volume_->IntegrateScan(image, pose_cuda, mask);

		cudaFree(mask);
	}
}
