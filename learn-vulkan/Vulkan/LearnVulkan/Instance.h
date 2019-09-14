#pragma once
#include "Config.h"

namespace LearnVulkan
{
	class PhysicalDevice;
	
	class Instance
	{
	public:
		Instance(const vk::ApplicationInfo& vAppInfo,
			const std::vector<const char*>& vLayers = Default::Instance::Layers,
			const std::vector<const char*>& vExtensions = Default::Instance::fetchExtensions());

		PhysicalDevice initPhysicalDevice(const std::map<std::string, vk::QueueFlags>& vRequestedQueueFamilies = Default::PhysicalDevice::RequestedQueueFamilies) const;
		
		vk::Instance& fetchInstance() { return m_Instance.get(); }

	private:
		void __createDebugCallback();
		
		vk::UniqueInstance m_Instance;
	};
}
