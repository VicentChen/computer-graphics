#include "Device.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
Device PhysicalDevice::initDevice(const std::vector<const char*>& vLayerNames, const std::vector<const char*>& vExtensionNames, const std::map<std::string, float>& vQueuePriorities) const
{
	std::map<std::string, float> QueuePriorities = vQueuePriorities;
	if (vQueuePriorities.size() == 0)
		for (const auto&[Name, Index] : m_QueueFamilyIndices)
			QueuePriorities[Name] = 1.0f;
	assert(QueuePriorities.size() == m_QueueFamilyIndices.size());
	
	std::vector<vk::DeviceQueueCreateInfo> QueueCreateInfos;
	for (const auto&[Name, Index] : m_QueueFamilyIndices)
		QueueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo{
			vk::DeviceQueueCreateFlags(),
			Index,
			1,
			&QueuePriorities[Name]
		});
	
	vk::DeviceCreateInfo Info = {
		vk::DeviceCreateFlags(),
		static_cast<uint32_t>(QueueCreateInfos.size()),
		QueueCreateInfos.data(),
		static_cast<uint32_t>(vLayerNames.size()),
		vLayerNames.data(),
		static_cast<uint32_t>(vExtensionNames.size()),
		vExtensionNames.data()
	};

	return Device(m_Device.createDeviceUnique(Info));
}
