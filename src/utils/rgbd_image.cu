// Copyright 2018 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#include "utils/rgbd_image.h"

namespace refusion {

	RgbdImage::~RgbdImage()
	{
		// Note from Connor: in the optical flow code, this funciton gets called every time the RgbdImage leaves any
		// scope (not just the scope it is defined in). This means memory is being released prematurely.
		// There is probably a "proper" fix for this, but having a manual memory management flag is a quick fix to allow
		// me to allocate memory outside of the class.
		if (manual_memory_management) { return; } // Don't continue if we are managing memory manually.
		cudaDeviceSynchronize();
		cudaFree(rgb_);
		cudaFree(depth_);
	}

	void RgbdImage::Init(const RgbdSensor &sensor)
	{
		Init(sensor, false);
	}

	void RgbdImage::Init(const RgbdSensor &sensor, bool manual_memory_management)
	{
		sensor_ = sensor;

		// Note from Connor: this class is unable to manage itself in the optical flow code, so we provide a mechanism
		// to perform manual memory management.
		this->manual_memory_management = manual_memory_management;
		if (manual_memory_management) { return; } // Don't continue if we are managing memory manually.

		cudaMallocManaged(&rgb_, sizeof(uchar3) * sensor_.rows * sensor.cols);
		cudaMallocManaged(&depth_, sizeof(float) * sensor_.rows * sensor.cols);
		cudaDeviceSynchronize();
	}

	__host__ __device__ inline float3 RgbdImage::GetPoint3d(int u, int v) const
	{
		float3 point;
		point.z = depth_[v * sensor_.cols + u];
		point.x = (static_cast<float>(u) - sensor_.cx) * point.z / sensor_.fx;
		point.y = (static_cast<float>(v) - sensor_.cy) * point.z / sensor_.fy;
		return point;
	}

	__host__ __device__ inline float3 RgbdImage::GetPoint3d(int i) const
	{
		int v = i / sensor_.cols;
		int u = i - sensor_.rows * v;
		return GetPoint3d(u, v);
	}

}  // namespace refusion
