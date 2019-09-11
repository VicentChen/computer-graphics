#pragma once

#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>

namespace LearnVulkan
{
	inline bool isVkSuccess(VkResult vResult) { return vResult == VK_SUCCESS; }
}