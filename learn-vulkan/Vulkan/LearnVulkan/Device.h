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
			vk::PhysicalDeviceFeatures vFeatures = Default::PhysicalDevice::getFeatures(),
			const std::map<std::string, float>& vQueuePriorities = {});

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
	class Buffer;
	class CommandPool;
	class DescriptorPool;
	class Image;
	
	class Device
	{
	public:
		vk::Device& fetchDevice() { return m_Device.get(); }

		Queue initQueue(const std::string& vName, uint32_t vCount = 1);
		Swapchain initSwapchain(PhysicalDevice& vPhysicalDevice, Surface& vSurface, uint32_t vWidth = Default::Window::WIDTH, uint32_t vHeight = Default::Window::HEIGHT, vk::ImageViewCreateInfo vImageViewCreateInfo = Default::Swapchain::ImageViewCreateInfo);
		DescriptorPool initDescriptorPool();
		Shader initShader(vk::ShaderStageFlagBits vStage, const std::string& vShaderFilePath, const std::vector<vk::VertexInputBindingDescription>& vBindings = {}, const std::vector<vk::VertexInputAttributeDescription>& vAttributes = {}, const std::string& vEntrance = Default::Shader::Entrance);
		Buffer initBuffer(CommandPool* vCommandPool, Queue* vGraphicsQueue, const void* vData, uint32_t vSize, vk::BufferUsageFlags vUsageFlags, vk::MemoryPropertyFlags vPropertyFlags);
		Image initImage(const std::string& vPath, vk::Format vFormat, vk::SampleCountFlagBits vSampleFlag, vk::ImageTiling vTiling, vk::ImageUsageFlags vUsageFlag, CommandPool* vCommandPool);
		
		void transferData(const void* vData, uint32_t vSize, Buffer& vBuffer);

		void setSwapchainPtr(Swapchain* vSwapchain) { m_pSwapchain = vSwapchain; }
		
		static Device createByPhysicalDevice(PhysicalDevice& vPhysicalDevice,
			const std::vector<const char*>& vLayerNames = Default::PhysicalDevice::LayerNames,
			const std::vector<const char*>& vExtensionNames = Default::PhysicalDevice::ExtensionNames,
			vk::PhysicalDeviceFeatures vFeatures = Default::PhysicalDevice::getFeatures(),
			const std::map<std::string, float>& vQueuePriorities = {}) { return vPhysicalDevice.initDevice(vLayerNames, vExtensionNames, vFeatures, vQueuePriorities); }

	private:
		Buffer __createBuffer(const void* vData, uint32_t vSize, vk::BufferUsageFlags vUsageFlags, vk::MemoryPropertyFlags vPropertyFlags);
		Device(vk::UniqueDevice&& vDevice, const std::map<std::string, uint32_t>& vQueueFamilyIndices, PhysicalDevice* vPhysicalDevice) : m_Device(std::move(vDevice)), m_QueueFamilyIndices(vQueueFamilyIndices), m_pPhysicalDevice(vPhysicalDevice) {}

		vk::UniqueDevice m_Device;
		std::map<std::string, uint32_t> m_QueueFamilyIndices;
		Swapchain* m_pSwapchain = nullptr;
		PhysicalDevice* m_pPhysicalDevice = nullptr;
		
		friend class PhysicalDevice;
	};
}