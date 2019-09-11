#pragma once
#include "Common.h"

namespace LearnVulkan
{
	namespace Default
	{
		namespace Window
		{
			const int WIDTH = 800;
			const int HEIGHT = 600;
			const std::string TITLE = "Learn Vulkan";
		}

		namespace Application
		{
			const vk::ApplicationInfo Info = {
				"Learn Vulkan",
				0,
				"Learn Vulkan Engine",
				0,
				VK_API_VERSION_1_1
			};
			
			inline vk::ApplicationInfo fetchInfo() { return vk::ApplicationInfo(Application::Info); }
		}

		namespace Instance
		{
			inline const char* Extensions[] = {
				"VK_LAYER_LUNARG_api_dump",
				"VK_LAYER_LUNARG_monitor"
			};
		}
	}
}