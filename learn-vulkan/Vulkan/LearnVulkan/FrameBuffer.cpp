#include "FrameBuffer.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
void FrameBuffer::constructFrameBuffer()
{
	auto& Attachments = m_pSwapchain->fetchImageViews();

	for (auto& Attachment : Attachments)
	{
		vk::FramebufferCreateInfo Info = {
			vk::FramebufferCreateFlags(),
			m_pRenderPass->fetchRenderPass(),
			1,
			&(Attachment.get()),
			m_pSwapchain->getExtent().width,
			m_pSwapchain->getExtent().height,
			1
		};

		m_FrameBuffers.emplace_back(m_pDevice->fetchDevice().createFramebufferUnique(Info));
	}
}
