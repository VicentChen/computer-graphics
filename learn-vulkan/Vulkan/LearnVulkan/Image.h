#pragma once
#include "Config.h"
#include "Device.h"
#include "Buffer.h"
#include "CommandPool.h"

namespace LearnVulkan
{
	class Image
	{
	public:
		Image(vk::UniqueImage&& vImage, vk::UniqueDeviceMemory&& vMemory, int vWidth, int vHeight, int vChannels, CommandPool* vCommandPool, Device* vDevice) :
			m_Image(std::move(vImage)),
			m_Memory(std::move(vMemory)),
			m_Width(vWidth),
			m_Height(vHeight),
			m_Channels(vChannels),
			m_Size(vWidth * vHeight * vChannels),
			m_pCommandPool(vCommandPool),
			m_pDevice(vDevice) { }

		void transitionImageLayout(vk::PipelineStageFlags vSrcStage, vk::ImageLayout vOldLayout, vk::PipelineStageFlags vDstStage, vk::ImageLayout vNewLayout);
		void copyBuffer2Image(Buffer& vBuffer);

		vk::Image fetchImage() { return m_Image.get(); }
		vk::ImageView fetchImageView() { return m_ImageView.get(); }
		vk::Sampler fetchSampler() { return m_Sampler.get(); }
		
	private:
		vk::UniqueImage m_Image;
		vk::UniqueImageView m_ImageView;
		vk::UniqueSampler m_Sampler;
		vk::UniqueDeviceMemory m_Memory;

		CommandPool* m_pCommandPool = nullptr;
		Device* m_pDevice = nullptr;

		int m_Width = 0;
		int m_Height = 0;
		int m_Channels = 0;
		int m_Size = 0;

		friend class Device;
	};
}