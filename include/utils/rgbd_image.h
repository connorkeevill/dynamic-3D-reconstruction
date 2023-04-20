// Copyright 2018 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>
#include "utils/rgbd_sensor.h"

namespace refusion {

/**
 * @brief      Class representing an RGB-D image.
 */
	class RgbdImage
	{
	public:
		/**
		 * @brief      Destroys the object.
		 */
		~RgbdImage();

		/**
		 * @brief      Initializes the class by allocating the necessary memory for
		 *             the images.
		 *
		 * @param[in]  sensor  The intrinsic parameters of the RGB-D sensor
		 */
		void Init(const RgbdSensor &sensor);

		/**
		 * @brief      Initializes the class by allocating the necessary memory for
		 * 				the images. Also allows us to specify that manual memory is wanted.
		 * @param sensor
		 * @param manual_memory_management
		 */
		void Init(const RgbdSensor &sensor, bool manual_memory_management);

		/**
		 * @brief      Gets the 3D point corresponding to the given pixel.
		 *
		 * @param[in]  u     Horizontal coordinate of the pixel
		 * @param[in]  v     Vertcal coordinate of the pixel
		 *
		 * @return     The 3D point.
		 */
		__host__ __device__

		float3 GetPoint3d(int u, int v) const;

		/**
		 * @brief      Gets the 3D point corresponding to the given pixel.
		 *
		 * @param[in]  i     The linear index of the pixel
		 *
		 * @return     The 3D point.
		 */
		__host__ __device__

		float3 GetPoint3d(int i) const;


		/** The RGB image */
		uchar3 *rgb_;

		/** The depth image */
		float *depth_;

		/** The intrinsic parameters of the RGB-D sensor */
		RgbdSensor sensor_;

		/** Whether or not to perform manual memory management (manual here meaning that memory is managed outside of the
		 * class. Automatic means that the RgbdImage will allocate and free memory itself. */
		bool manual_memory_management = false;
	};

}  // namespace refusion
