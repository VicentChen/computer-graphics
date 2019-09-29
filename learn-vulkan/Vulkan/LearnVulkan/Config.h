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

			const int FRAMES_IN_FLIGHT = 2;
		}

		namespace Application
		{
			const vk::ApplicationInfo Info = {
				"Learn Vulkan",
				0,
				"Learn Vulkan Engine",
				VK_MAKE_VERSION(1, 0, 0),
				VK_API_VERSION_1_0
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

				for (uint32_t i = 0; i < GLFWExtensionCount; i++)
					Extensions.emplace_back(GLFWExtensions[i]);
				
				return Extensions;
			}

			const vk::DebugUtilsMessageSeverityFlagsEXT MessageSeverityFlags(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
			const vk::DebugUtilsMessageTypeFlagsEXT MessageTypeFlags(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
		}

		namespace PhysicalDevice
		{
			const std::string GRAPHICS_QUEUE_NAME = "GraphicsQueue";
			const std::string COMPUTE_QUEUE_NAME = "ComputeQueue";
			const std::string TRANSFER_QUEUE_NAME = "TransferQueue";
			const std::string PRESENT_QUEUE_NAME = "PresentQueue";
			
			const std::map<std::string, vk::QueueFlags> RequestedQueueFamilies = {
				std::make_pair(GRAPHICS_QUEUE_NAME, vk::QueueFlags(vk::QueueFlagBits::eGraphics))
			};

			const std::vector<const char*> LayerNames = {
#ifndef NDEBUG
				"VK_LAYER_LUNARG_standard_validation",
#endif
			};
			const std::vector<const char*> ExtensionNames = {
				VK_KHR_SWAPCHAIN_EXTENSION_NAME
			};
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
			const vk::SemaphoreCreateInfo SemaphoreInfo = {};
			const vk::FenceCreateInfo FenceInfo = { vk::FenceCreateFlagBits::eSignaled };
		}

		namespace Swapchain
		{
			const vk::SurfaceFormatKHR Format = { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
			const vk::PresentModeKHR PresentMode = vk::PresentModeKHR::eFifo;
			const vk::ImageSubresourceRange SubresourceRange = {
				vk::ImageAspectFlagBits::eColor,
				0,
				1,
				0,
				1
			};
			const vk::ImageViewCreateInfo ImageViewCreateInfo = {
				vk::ImageViewCreateFlags(),
				vk::Image(),
				vk::ImageViewType::e2D,
				Format.format,
				vk::ComponentMapping(vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,vk::ComponentSwizzle::eIdentity,vk::ComponentSwizzle::eIdentity),
				SubresourceRange
			};
		}

		namespace Shader
		{
			const std::vector<glm::vec3> Vertices = {
				{ 0.0f, -0.5f, 0.0f },
				{ 0.5f,  0.5f, 0.0f },
				{-0.5f,  0.5f, 0.0f }
			};

			const std::vector<glm::vec3> Colors = {
				{ 1.0f, 0.0f, 0.0f },
				{ 0.0f, 1.0f, 0.0f },
				{ 0.0f, 0.0f, 1.0f }
			};

			// Bindings: for different vertex objects
			const std::vector<vk::VertexInputBindingDescription> VertexInputBindings = {
				vk::VertexInputBindingDescription { 0, sizeof(glm::vec3), vk::VertexInputRate::eVertex },
				vk::VertexInputBindingDescription { 1, sizeof(glm::vec3), vk::VertexInputRate::eVertex }
			};

			// Attributes : for different data(location) in same object
			const std::vector<vk::VertexInputAttributeDescription> VertexInputAttributes = {
				vk::VertexInputAttributeDescription { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
				vk::VertexInputAttributeDescription { 1, 1, vk::Format::eR32G32B32Sfloat, 0 }
			};
			
			const std::string VertexPath   = "Shaders/Triangle.vert.spv";
			const std::string FragmentPath = "Shaders/Triangle.frag.spv";
			const std::string Entrance = "main";
		}

		namespace RenderPass
		{
			const vk::ClearColorValue BLACK = { std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f } };
			const vk::ClearColorValue WHITE = { std::array<float, 4>{ 1.0f, 1.0f, 1.0f, 1.0f } };
		}
	}
}