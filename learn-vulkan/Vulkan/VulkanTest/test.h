#pragma once
#include <vulkan/vulkan.hpp>
#include <map>
#include <vector>

namespace PhysicalDevice
{
	inline const std::map<std::string, vk::QueueFlags> RequestedQueueFamilies = {
		std::make_pair(std::string("GraphicsQueue"), vk::QueueFlags(vk::QueueFlagBits::eGraphics))
	};
}
