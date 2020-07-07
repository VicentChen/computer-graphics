#include "Image.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
void Image::transitionImageLayout(vk::PipelineStageFlags vSrcStage, vk::ImageLayout vOldLayout, vk::PipelineStageFlags vDstStage, vk::ImageLayout vNewLayout)
{
	SingleCommand Command = m_pCommandPool->createSingleCommand();
	vk::CommandBuffer CommandBuffer = Command.fetchCommandBuffer();

	vk::ImageMemoryBarrier ImageBarrier = {
		vk::AccessFlags(),
		vk::AccessFlags(),
		vOldLayout,
		vNewLayout,
		VK_QUEUE_FAMILY_IGNORED,
		VK_QUEUE_FAMILY_IGNORED,
		m_Image.get(),
	{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};

	CommandBuffer.pipelineBarrier(vSrcStage, vDstStage, vk::DependencyFlags(), nullptr, nullptr, ImageBarrier);
}

//*********************************************************************
//FUNCTION:
void Image::copyBuffer2Image(Buffer& vBuffer)
{
	SingleCommand Command = m_pCommandPool->createSingleCommand();
	vk::CommandBuffer CommandBuffer = Command.fetchCommandBuffer();

	vk::BufferImageCopy CopyRegion = {
		0, 0, 0,
		vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
		vk::Offset3D(0, 0 , 0),
		vk::Extent3D(m_Width, m_Height, 1)
	};

	CommandBuffer.copyBufferToImage(vBuffer.fetchBuffer(), m_Image.get(), vk::ImageLayout::eTransferDstOptimal, CopyRegion);
}
