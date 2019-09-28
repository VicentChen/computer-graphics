#include "Device.h"
#include "Queue.h"
#include "Swapchain.h"
#include "Shader.h"
#include "Buffer.h"
#include <algorithm>

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
Device PhysicalDevice::initDevice(const std::vector<const char*>& vLayerNames, const std::vector<const char*>& vExtensionNames, const std::map<std::string, float>& vQueuePriorities)
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

	return Device(m_Device.createDeviceUnique(Info), m_QueueFamilyIndices, this);
}

//*********************************************************************
//FUNCTION:
Queue Device::initQueue(const std::string& vName, uint32_t vCount)
{
	uint32_t QueueFamilyIndex = 0;
	if (m_QueueFamilyIndices.find(vName) != m_QueueFamilyIndices.end()) QueueFamilyIndex = m_QueueFamilyIndices[vName];
	else VERBOSE_EXIT(std::string("Not supported name: ") + vName);

	vk::Queue VkQueue = m_Device->getQueue(QueueFamilyIndex, 0);
	return Queue(vName, VkQueue, QueueFamilyIndex);
}

//*********************************************************************
//FUNCTION:
Swapchain Device::initSwapchain(PhysicalDevice& vPhysicalDevice, Surface& vSurface, uint32_t vWidth, uint32_t vHeight, vk::ImageViewCreateInfo vImageViewCreateInfo)
{
	std::vector<vk::SurfaceFormatKHR> Formats = vPhysicalDevice.fetchPhysicalDevice().getSurfaceFormatsKHR(vSurface.fetchSurface().get());
	if (std::find(Formats.begin(), Formats.end(), Default::Swapchain::Format) == Formats.end()) VERBOSE_EXIT("not supported format");
	vk::SurfaceCapabilitiesKHR Capabilities = vPhysicalDevice.fetchPhysicalDevice().getSurfaceCapabilitiesKHR(vSurface.fetchSurface().get());
	vk::Extent2D Extent = {
		std::clamp(vWidth, Capabilities.minImageExtent.width, Capabilities.maxImageExtent.width),
		std::clamp(vHeight, Capabilities.minImageExtent.height, Capabilities.maxImageExtent.height) };

	std::vector<uint32_t> QueueFamilyIndices;
	for (const auto&[Name, Index] : m_QueueFamilyIndices) QueueFamilyIndices.emplace_back(Index);
	
	vk::SwapchainCreateInfoKHR Info = {
		vk::SwapchainCreateFlagsKHR(),
		vSurface.fetchSurface().get(),
		Capabilities.minImageCount,
		Default::Swapchain::Format.format,
		Default::Swapchain::Format.colorSpace,
		Extent,
		1,
		vk::ImageUsageFlagBits::eColorAttachment,
		vk::SharingMode::eExclusive,
		CAST_U32I(QueueFamilyIndices.size()),
		QueueFamilyIndices.data(),
		Capabilities.currentTransform,
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		Default::Swapchain::PresentMode,
		true,
		nullptr
	};

	Swapchain Result = Swapchain(m_Device->createSwapchainKHRUnique(Info), &vSurface, &vPhysicalDevice, this);
	Result.m_SwapchainImages = m_Device->getSwapchainImagesKHR(Result.m_Swapchain.get());
	for (const auto& Image : Result.m_SwapchainImages)
	{
		vImageViewCreateInfo.image = Image;
		Result.m_SwapchainImageViews.emplace_back(m_Device->createImageViewUnique(vImageViewCreateInfo));
	}
	Result.m_Extent = Extent;
	Result.m_Format = Default::Swapchain::Format;
	return Result;
}

//*********************************************************************
//FUNCTION:
Shader Device::initShader(vk::ShaderStageFlagBits vStage, const std::string& vShaderFilePath, const std::vector<vk::VertexInputBindingDescription>& vBindings, const std::vector<vk::VertexInputAttributeDescription>& vAttributes, const std::string& vEntrance)
{
	std::vector<char> ShaderFile = readFile(vShaderFilePath);
	vk::ShaderModuleCreateInfo ShaderInfo = {
		vk::ShaderModuleCreateFlags(),
		ShaderFile.size(),
		reinterpret_cast<const uint32_t*>(ShaderFile.data())
	};
	auto ShaderModule = m_Device->createShaderModuleUnique(ShaderInfo);
	vk::PipelineShaderStageCreateInfo ShaderStageInfo = {
		vk::PipelineShaderStageCreateFlags(),
		vStage,
		ShaderModule.get(),
		vEntrance.c_str()
	};
	
	return Shader(std::move(ShaderModule), vBindings, vAttributes, ShaderStageInfo);
}

//*********************************************************************
//FUNCTION:
Buffer Device::initVertexBuffer(const vk::BufferCreateInfo& vInfo, const void* vData, uint32_t vSize)
{
	vk::UniqueBuffer VertexBuffer = m_Device->createBufferUnique(vInfo);
	vk::MemoryRequirements MemoryRequirements = m_Device.get().getBufferMemoryRequirements(VertexBuffer.get());
	vk::PhysicalDeviceMemoryProperties MemoryProperties = m_pPhysicalDevice->fetchPhysicalDevice().getMemoryProperties();
	vk::MemoryAllocateInfo MemoryAllocateInfo = { MemoryRequirements.size };
	for (uint32_t i = 0; i < MemoryProperties.memoryTypeCount; i++)
	{
		if ((MemoryRequirements.memoryTypeBits & (1 << i)) && (MemoryProperties.memoryTypes[i].propertyFlags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)))
		{
			MemoryAllocateInfo.memoryTypeIndex = i;
			break;
		}
	}
	vk::UniqueDeviceMemory VertexMemory = m_Device->allocateMemoryUnique(MemoryAllocateInfo);
	return Buffer(std::move(VertexBuffer), std::move(VertexMemory), vData, vSize, this);
}
