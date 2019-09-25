#pragma once
#include "Surface.h"

namespace LearnVulkan
{
	class Device;
	
	class PhysicalDevice
	{
	public:
		Device initDevice(const std::vector<const char*>& vLayerNames = Default::PhysicalDevice::LayerNames,
			const std::vector<const char*>& vExtensionNames = Default::PhysicalDevice::ExtensionNames,
			const std::map<std::string, float>& vQueuePriorities = {}) const;

		std::vector<vk::SurfaceFormatKHR> getSurfaceFormats(Surface& vSurface) const { return m_Device.getSurfaceFormatsKHR(vSurface.fetchSurface().get()); }

		vk::PhysicalDevice& fetchPhysicalDevice() { return m_Device; }
		uint32_t getQueueFamilyIndex(const std::string& vQueuName) const { return m_QueueFamilyIndices.find(vQueuName)->second; }
		
		static PhysicalDevice createByInstance(Instance& vInstance, Surface& vSurface, const std::map<std::string, vk::QueueFlags>& vRequestedQueueFamilies = Default::PhysicalDevice::RequestedQueueFamilies) { return vInstance.initPhysicalDevice(vSurface, vRequestedQueueFamilies); }

	private:
		PhysicalDevice(vk::PhysicalDevice& vDevice, const std::map<std::string, uint32_t>& vQueueFamilyIndices) : m_Device(vDevice), m_QueueFamilyIndices(vQueueFamilyIndices) {}

		vk::PhysicalDevice m_Device;
		std::map<std::string, uint32_t> m_QueueFamilyIndices;
		
		friend class Instance;
	};

	class Queue;
	class Swapchain;
	class Shader;
	
	class Device
	{
	public:
		vk::Device& fetchDevice() { return m_Device.get(); }

		Queue initQueue(const std::string& vName, uint32_t vCount = 1);
		Swapchain initSwapchain(PhysicalDevice& vPhysicalDevice, Surface& vSurface, uint32_t vWidth = Default::Window::WIDTH, uint32_t vHeight = Default::Window::HEIGHT, vk::ImageViewCreateInfo vImageViewCreateInfo = Default::Swapchain::ImageViewCreateInfo);
		Shader initShader(vk::ShaderStageFlagBits vStage, const std::string& vShaderFilePath, const std::vector<vk::VertexInputBindingDescription>& vBindings = {}, const std::vector<vk::VertexInputAttributeDescription>& vAttributes = {}, const std::string& vEntrance = Default::Pipeline::Entrance);

		vk::UniqueSemaphore initSemaphore(const vk::SemaphoreCreateInfo& vInfo = Default::Device::SemaphoreInfo) { return m_Device->createSemaphoreUnique(vInfo); }
		
		static Device createByPhysicalDevice(const PhysicalDevice& vPhysicalDevice,
			const std::vector<const char*>& vLayerNames = Default::PhysicalDevice::LayerNames,
			const std::vector<const char*>& vExtensionNames = Default::PhysicalDevice::ExtensionNames,
			const std::map<std::string, float>& vQueuePriorities = {}) { return vPhysicalDevice.initDevice(vLayerNames, vExtensionNames, vQueuePriorities); }

	private:
		Device(vk::UniqueDevice&& vDevice, const std::map<std::string, uint32_t>& vQueueFamilyIndices) : m_Device(std::move(vDevice)), m_QueueFamilyIndices(vQueueFamilyIndices) {}

		vk::UniqueDevice m_Device;
		std::map<std::string, uint32_t> m_QueueFamilyIndices;
		
		friend class PhysicalDevice;
	};
}