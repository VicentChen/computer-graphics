#include "Device.h"
#include "Queue.h"
#include "Swapchain.h"
#include "Shader.h"
#include "Buffer.h"
#include "CommandPool.h"
#include "DescriptorPool.h"
#include "Image.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <algorithm>

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
Device PhysicalDevice::initDevice(const std::vector<const char*>& vLayerNames, const std::vector<const char*>& vExtensionNames, vk::PhysicalDeviceFeatures vFeatures, const std::map<std::string, float>& vQueuePriorities)
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
		vExtensionNames.data(),
		&vFeatures
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
DescriptorPool Device::initDescriptorPool()
{
	auto SwapchainSize = CAST_U32I(m_pSwapchain->fetchImageViews().size());
	std::vector<vk::DescriptorPoolSize> Sizes;
	Sizes.emplace_back(vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, SwapchainSize });
	Sizes.emplace_back(vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, SwapchainSize });

	vk::DescriptorPoolCreateInfo PoolCreateInfo = {
		vk::DescriptorPoolCreateFlags(),
		SwapchainSize,
		CAST_U32I(Sizes.size()),
		Sizes.data()
	};

	DescriptorPool Pool(m_Device->createDescriptorPoolUnique(PoolCreateInfo), this, m_pSwapchain);
	return Pool;
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
Buffer Device::initBuffer(CommandPool* vCommandPool, Queue* vGraphicsQueue, const void* vData, uint32_t vSize, vk::BufferUsageFlags vUsageFlags, vk::MemoryPropertyFlags vPropertyFlags)
{
	Buffer buffer = __createBuffer(vData, vSize, vUsageFlags, vPropertyFlags);
	if (vData)
	{
		if (vPropertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)
		{
			transferData(vData, vSize, buffer);
		}
		else
		{
			Buffer StagingBuffer = __createBuffer(vData, vSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
			transferData(vData, vSize, StagingBuffer);
			vk::CommandBufferAllocateInfo AllocateInfo = {
				vCommandPool->fetchCommandPool(),
				vk::CommandBufferLevel::ePrimary,
				1
			};

			std::vector<vk::UniqueCommandBuffer> UniqueCommandBuffers = m_Device->allocateCommandBuffersUnique(AllocateInfo);
			vk::UniqueCommandBuffer UniqueCommandBuffer = std::move(UniqueCommandBuffers[0]);
			vk::CommandBuffer CommandBuffer = UniqueCommandBuffer.get();
			vk::CommandBufferBeginInfo BeginInfo = { vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
			CommandBuffer.begin(BeginInfo);
			vk::BufferCopy CopyRegion = { 0, 0, buffer.size() };
			CommandBuffer.copyBuffer(StagingBuffer.fetchBuffer(), buffer.fetchBuffer(), CopyRegion);
			CommandBuffer.end();

			vk::SubmitInfo SubmitInfo = {
				0, nullptr, nullptr,
				1, &CommandBuffer,
				0, nullptr
			};
			vGraphicsQueue->fetchQueue().submit(SubmitInfo, vk::Fence(nullptr));
			vGraphicsQueue->fetchQueue().waitIdle();
		}
	}
	return buffer;
}

//*********************************************************************
//FUNCTION:
Image Device::initImage(const std::string& vPath, vk::Format vFormat, vk::SampleCountFlagBits vSampleFlag, vk::ImageTiling vTiling, vk::ImageUsageFlags vUsageFlag, CommandPool* vCommandPool)
{
	int Width, Height, Channels;
	stbi_uc* Pixels = stbi_load(vPath.c_str(), &Width, &Height, &Channels, STBI_rgb_alpha);
	if (!Pixels) VERBOSE_EXIT("Unable to read image");
	int ImageSize = Width * Height * 4;
	Buffer StagingBuffer = __createBuffer(Pixels, ImageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	vk::ImageCreateInfo Info = {
		vk::ImageCreateFlags(),
		vk::ImageType::e2D,
		vFormat,
		vk::Extent3D(Width, Height, 1),
		1,
		1,
		vSampleFlag,
		vTiling,
		vUsageFlag,
		vk::SharingMode::eExclusive
	};
	vk::UniqueImage image = m_Device->createImageUnique(Info);
	
	// allocate memory
	vk::MemoryRequirements MemoryRequirements = m_Device.get().getImageMemoryRequirements(image.get());
	vk::PhysicalDeviceMemoryProperties MemoryProperties = m_pPhysicalDevice->fetchPhysicalDevice().getMemoryProperties();
	vk::MemoryAllocateInfo MemoryAllocateInfo = { MemoryRequirements.size };
	for (uint32_t i = 0; i < MemoryProperties.memoryTypeCount; i++)
	{
		if ((MemoryRequirements.memoryTypeBits & (1 << i)) && (MemoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
		{
			MemoryAllocateInfo.memoryTypeIndex = i;
			break;
		}
	}
	vk::UniqueDeviceMemory Memory = m_Device->allocateMemoryUnique(MemoryAllocateInfo);
	m_Device->bindImageMemory(image.get(), Memory.get(), 0);

	Image RealImage = Image(std::move(image), std::move(Memory), Width, Height, 4, vCommandPool, this);
	RealImage.transitionImageLayout(vk::PipelineStageFlagBits::eTopOfPipe, vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits::eTransfer, vk::ImageLayout::eTransferDstOptimal);
	RealImage.copyBuffer2Image(StagingBuffer);
	RealImage.transitionImageLayout(vk::PipelineStageFlagBits::eTransfer, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eFragmentShader, vk::ImageLayout::eShaderReadOnlyOptimal);

	vk::ImageViewCreateInfo ViewCreateInfo = {
		vk::ImageViewCreateFlags(),
		RealImage.m_Image.get(),
		vk::ImageViewType::e2D,
		vFormat,
		vk::ComponentMapping(),
		{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	RealImage.m_ImageView = m_Device->createImageViewUnique(ViewCreateInfo);

	vk::SamplerCreateInfo SamplerCreateInfo = {
		vk::SamplerCreateFlags(),
		vk::Filter::eLinear,
		vk::Filter::eLinear,
		vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat,
		vk::SamplerAddressMode::eRepeat,
		vk::SamplerAddressMode::eRepeat,
		0.0f,
		true,
		16.0f,
		false,
		vk::CompareOp::eAlways,
		0.0f,
		0.0f,
		vk::BorderColor::eIntOpaqueBlack,
		false
	};

	RealImage.m_Sampler = m_Device->createSamplerUnique(SamplerCreateInfo);
	
	return RealImage;
}

//*********************************************************************
//FUNCTION:
Buffer Device::__createBuffer(const void* vData, uint32_t vSize, vk::BufferUsageFlags vUsageFlags, vk::MemoryPropertyFlags vPropertyFlags)
{
	// create buffer
	vk::BufferCreateInfo CreateInfo = {
		vk::BufferCreateFlags(),
		vSize,
		vUsageFlags
	};
	vk::UniqueBuffer buffer = m_Device->createBufferUnique(CreateInfo);

	// allocate memory
	vk::MemoryRequirements MemoryRequirements = m_Device.get().getBufferMemoryRequirements(buffer.get());
	vk::PhysicalDeviceMemoryProperties MemoryProperties = m_pPhysicalDevice->fetchPhysicalDevice().getMemoryProperties();
	vk::MemoryAllocateInfo MemoryAllocateInfo = { MemoryRequirements.size };
	for (uint32_t i = 0; i < MemoryProperties.memoryTypeCount; i++)
	{
		if ((MemoryRequirements.memoryTypeBits & (1 << i)) && (MemoryProperties.memoryTypes[i].propertyFlags & vPropertyFlags) == vPropertyFlags)
		{
			MemoryAllocateInfo.memoryTypeIndex = i;
			break;
		}
	}
	vk::UniqueDeviceMemory Memory = m_Device->allocateMemoryUnique(MemoryAllocateInfo);
	m_Device->bindBufferMemory(buffer.get(), Memory.get(), 0);
	return Buffer(std::move(buffer), std::move(Memory), vData, vSize, this);
}

//*********************************************************************
//FUNCTION:
void Device::transferData(const void* vData, uint32_t vSize, Buffer& vBuffer)
{
	void* Data;
	vkMapMemory(m_Device.get(), vBuffer.fetchMemory(), 0, vSize, 0, &Data);
	memcpy(Data, vData, vSize);
	vkUnmapMemory(m_Device.get(), vBuffer.fetchMemory());
}
