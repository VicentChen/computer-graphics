#include "Instance.h"
#include "Surface.h"
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
#ifndef NDEBUG
	__createDebugCallback();
#endif
}

Instance::~Instance()
{
#ifndef NDEBUG
	m_Instance->destroyDebugUtilsMessengerEXT(m_DebugMessenger, nullptr, vk::DispatchLoaderDynamic{ m_Instance.get() });
#endif
}

//*********************************************************************
//FUNCTION:
PhysicalDevice Instance::initPhysicalDevice(Surface& vSurface, const std::map<std::string, vk::QueueFlags>& vRequestedQueueFamilies) const
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
			bool Found = false;
			size_t QueueFamilyIndex = 0;
			for(auto i = QueueProperties.begin(); !Found && i != QueueProperties.end(); i = QueueProperties.begin() + QueueFamilyIndex)
			{
				QueueFamilyIndex = std::distance(QueueProperties.begin(),
					std::find_if(i, QueueProperties.end(),
						[&](vk::QueueFamilyProperties const& vProperty) { return QueueFlag == (vProperty.queueFlags & QueueFlag); }));
				if (!Device.getSurfaceSupportKHR(QueueFamilyIndex, vSurface.fetchSurface().get())) continue;
#ifdef _WIN32 || _WIN64
				if (!Device.getWin32PresentationSupportKHR(QueueFamilyIndex)) continue;
#endif
				if (QueueFamilyIndex != QueueProperties.size())
				{
					QueueFamilyIndices.insert(std::make_pair(QueueName, QueueFamilyIndex));
					Found = true;
				}
			}
			if (!Found) break; //early end
		}
		if (QueueFamilyIndices.size() != vRequestedQueueFamilies.size())
		{
			LOG_INFO(std::string("Device ") + Device.getProperties().deviceName + " not suitable");
			continue;
		}

		vk::PhysicalDeviceFeatures SupportedFeatures = Device.getFeatures();
		if (!SupportedFeatures.samplerAnisotropy)
		{
			LOG_INFO(std::string("Device ") + Device.getProperties().deviceName + " not suitable");
			continue;
		}
		
		SuitableDevices.emplace_back(PhysicalDevice(Device, QueueFamilyIndices));
	}
	if (SuitableDevices.empty()) VERBOSE_EXIT("No suitable device");
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
	m_DebugMessenger = m_Instance->createDebugUtilsMessengerEXT(CreateInfo, nullptr, vk::DispatchLoaderDynamic{ m_Instance.get() });
}
