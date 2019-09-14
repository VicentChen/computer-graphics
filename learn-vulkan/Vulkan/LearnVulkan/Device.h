#pragma once
#include "Instance.h"

namespace LearnVulkan
{
	class Device;

	class PhysicalDevice
	{
	public:
		Device initDevice(const std::vector<const char*>& vLayerNames = Default::PhysicalDevice::LayerNames,
			const std::vector<const char*>& vExtensionNames = Default::PhysicalDevice::ExtensionNames,
			const std::map<std::string, float>& vQueuePriorities = {}) const;
		
		vk::PhysicalDevice& fetchPhysicalDevice() { return m_Device; }
		uint32_t getQueueFamilyIndex(const std::string& vQueuName) const { return m_QueueFamilyIndices.find(vQueuName)->second; }
		
		static PhysicalDevice createByInstance(Instance& vInstance, const std::map<std::string, vk::QueueFlags>& vRequestedQueueFamilies = Default::PhysicalDevice::RequestedQueueFamilies) { return vInstance.initPhysicalDevice(vRequestedQueueFamilies); }

private:
		PhysicalDevice(vk::PhysicalDevice& vDevice, const std::map<std::string, uint32_t>& vQueueFamilyIndices) : m_Device(vDevice), m_QueueFamilyIndices(vQueueFamilyIndices) {}

		vk::PhysicalDevice m_Device;
		std::map<std::string, uint32_t> m_QueueFamilyIndices;
		
		friend class Instance;
	};

	class Device
	{
	public:
		vk::Device& fetchDevice() { return m_Device.get(); }
		
		static Device createByPhysicalDevice(const PhysicalDevice& vPhysicalDevice,
			const std::vector<const char*>& vLayerNames = Default::PhysicalDevice::LayerNames,
			const std::vector<const char*>& vExtensionNames = Default::PhysicalDevice::ExtensionNames,
			const std::map<std::string, float>& vQueuePriorities = {}) { return vPhysicalDevice.initDevice(vLayerNames, vExtensionNames, vQueuePriorities); }

	private:
		Device(vk::UniqueDevice&& vDevice) : m_Device(std::move(vDevice)) {}
		
		vk::UniqueDevice m_Device;
		
		friend class PhysicalDevice;
	};
}