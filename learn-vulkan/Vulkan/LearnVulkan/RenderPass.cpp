#include "RenderPass.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
void RenderPass::constructRenderPass()
{
	vk::AttachmentDescription ColorAttachment = {
		vk::AttachmentDescriptionFlags(),
		m_pSwapchain->getFormat().format,
		vk::SampleCountFlagBits::e1,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::ePresentSrcKHR
	};

	vk::AttachmentReference ColorAttachmentReference = {
		0,
		vk::ImageLayout::eColorAttachmentOptimal
	};

	vk::SubpassDescription Subpass = {
		vk::SubpassDescriptionFlags(),
		vk::PipelineBindPoint::eGraphics,
		0,
		nullptr,
		1,
		&ColorAttachmentReference
	};

	vk::SubpassDependency Dependency = {
	VK_SUBPASS_EXTERNAL,
	0,
	vk::PipelineStageFlagBits::eColorAttachmentOutput,
	vk::PipelineStageFlagBits::eColorAttachmentOutput,
	vk::AccessFlags(),
	vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite
	};

	vk::RenderPassCreateInfo Info = {
		vk::RenderPassCreateFlags(),
		1,
		&ColorAttachment,
		1,
		&Subpass,
		1,
		&Dependency
	};

	 m_RenderPass = m_pDevice->fetchDevice().createRenderPassUnique(Info);
}
