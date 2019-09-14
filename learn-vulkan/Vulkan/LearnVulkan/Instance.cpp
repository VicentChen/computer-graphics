#include "Instance.h"
#include "Device.h"

using namespace LearnVulkan;

Instance::Instance(const vk::ApplicationInfo& vAppInfo, const std::vector<const char*>& vLayers, const std::vector<const char*>& vExtensions)
{
	std::vector<const char*> Layers = vLayers;
	std::vector<const char*> Extensions = vExtensions;
	
	vk::InstanceCreateInfo Info = {
		vk::InstanceCreateFlags(),
		&vAppInfo,
		CAST_U32I(Layers.size()),
		Layers.data(),
		CAST_U32I(Extensions.size()),
		Extensions.data()
	};

	if(!checkInstanceExtensionSupport(m_Instance.get(), Layers, Extensions)) LOG_ERROR("Not supported extension");
	m_Instance = vk::createInstanceUnique(Info);
	__createDebugCallback();
}

//*********************************************************************
//FUNCTION:
PhysicalDevice Instance::initPhysicalDevice(const std::map<std::string, vk::QueueFlags>& vRequestedQueueFamilies) const
{
	std::vector<vk::PhysicalDevice> AvailableDevices = m_Instance->enumeratePhysicalDevices();
	std::vector<PhysicalDevice> SuitableDevices;
	std::map<std::string, uint32_t> QueueFamilyIndices;
	for (auto& Device : AvailableDevices)
	{
		QueueFamilyIndices.clear();
		auto QueueProperties = Device.getQueueFamilyProperties();
		for (const auto& [QueueName, QueueFlag] : vRequestedQueueFamilies)
		{
			size_t QueueFamilyIndex = std::distance(QueueProperties.begin(),
				std::find_if(QueueProperties.begin(), QueueProperties.end(),
					[&](vk::QueueFamilyProperties const& vProperty) { return QueueFlag == (vProperty.queueFlags & QueueFlag); }));
			if (QueueFamilyIndex != QueueProperties.size()) QueueFamilyIndices.insert(std::make_pair(QueueName, QueueFamilyIndex));
			else break; // early end
		}
		if (QueueFamilyIndices.size() != vRequestedQueueFamilies.size()) continue;
		
		SuitableDevices.emplace_back(PhysicalDevice(Device, QueueFamilyIndices));
	}

	return SuitableDevices.front();
}

//*********************************************************************
//FUNCTION:
void Instance::__createDebugCallback()
{
	vk::DebugUtilsMessengerCreateInfoEXT CreateInfo {
		vk::DebugUtilsMessengerCreateFlagsEXT(),
		Default::Instance::MessageSeverityFlags,
		Default::Instance::MessageTypeFlags,
		debugMessengerCallback,
		nullptr
	};
	[[maybe_unused]] PFN_vkVoidFunction CreateDebugFunc = m_Instance->getProcAddr("vkCreateDebugUtilsMessengerEXT");
	[[maybe_unused]] PFN_vkVoidFunction DestroyDebugFunc = m_Instance->getProcAddr("vkDestroyDebugUtilsMessengerEXT");
	m_Instance->createDebugUtilsMessengerEXTUnique(CreateInfo, nullptr, vk::DispatchLoaderDynamic{ m_Instance.get() });
}
