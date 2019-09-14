#pragma once
#include "Common.h"
#include <map>
#include <vector>
#include <string>

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
			
			inline vk::ApplicationInfo fetchInfo() { return vk::ApplicationInfo(Info); }
		}

		namespace Instance
		{
			const std::vector<const char*> Layers = {
#ifndef NDEBUG
				"VK_LAYER_LUNARG_standard_validation",
#endif
				"VK_LAYER_LUNARG_api_dump",
				"VK_LAYER_LUNARG_monitor",
			};

			inline const std::vector<const char*> fetchExtensions()
			{
				std::vector<const char*> Extensions;
				// lunarG extensions
#ifndef NDEBUG
				Extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

				// GLFW extensions
				uint32_t GLFWExtensionCount = 0;
				const char** GLFWExtensions = glfwGetRequiredInstanceExtensions(&GLFWExtensionCount);

				for (int i = 0; i < GLFWExtensionCount; i++)
					Extensions.emplace_back(GLFWExtensions[i]);
				
				return Extensions;
			}

			inline const vk::DebugUtilsMessageSeverityFlagsEXT MessageSeverityFlags(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
			inline const vk::DebugUtilsMessageTypeFlagsEXT MessageTypeFlags(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
		}

		namespace PhysicalDevice
		{
			const std::string GRAPHICS_QUEUE_NAME = "GraphicsQueue";
			const std::string COMPUTE_QUEUE_NAME = "ComputeQueue";
			const std::string TRANSFER_QUEUE_NAME = "TransferQueue";
			
			const std::map<std::string, vk::QueueFlags> RequestedQueueFamilies = {
				std::make_pair(GRAPHICS_QUEUE_NAME, vk::QueueFlags(vk::QueueFlagBits::eGraphics))
			};

			const std::vector<const char*> LayerNames = {};
			const std::vector<const char*> ExtensionNames = {};
		}

		namespace Device
		{
			const vk::DeviceCreateInfo Info = {
				vk::DeviceCreateFlags(),
				0,
				nullptr,
				0,
				nullptr,
				0,
				nullptr
			};
		}

		namespace Queue
		{
			inline const float Priorities = 1.0f;
		}
	}
}