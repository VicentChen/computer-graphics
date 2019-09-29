#include "CommandPool.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
void CommandPool::constructCommandPool()
{
	vk::CommandPoolCreateInfo Info = {
		vk::CommandPoolCreateFlags(),
		m_pGraphicsQueue->getIndex()
	};

	m_CommandPool = m_pDevice->fetchDevice().createCommandPoolUnique(Info);
}

//*********************************************************************
//FUNCTION:
void CommandPool::constructCommandBuffers(std::vector<Buffer*>& vVertexBuffers, Buffer* vIndexBuffer)
{	
	vk::CommandBufferAllocateInfo AllocateInfo = {
		m_CommandPool.get(),
		vk::CommandBufferLevel::ePrimary,
		CAST_U32I(m_pSwapchain->fetchImageViews().size()),
	};
	m_CommandBuffers = m_pDevice->fetchDevice().allocateCommandBuffersUnique(AllocateInfo);

	vk::ClearValue Value = { Default::RenderPass::BLACK };

	std::vector<vk::Buffer> VertexBuffers;
	std::vector<vk::DeviceSize> Offsets;
	for (auto pBuffer : vVertexBuffers)
	{
		VertexBuffers.emplace_back(pBuffer->fetchBuffer());
		Offsets.emplace_back(0);
	}
	
	for (int i = 0; i < m_CommandBuffers.size(); i++)
	{
		vk::CommandBufferBeginInfo CommandBufferBeginInfo;
		CommandBufferBeginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
		m_CommandBuffers[i]->begin(CommandBufferBeginInfo);
		
		vk::RenderPassBeginInfo RenderPassBeginInfo = {
			m_pRenderPass->fetchRenderPass(),
			m_pFrameBuffer->fetchFrameBufferAt(i),
			{{0, 0}, m_pSwapchain->getExtent()},
			1,
			&Value
		};
		m_CommandBuffers[i]->beginRenderPass(RenderPassBeginInfo, vk::SubpassContents::eInline);

		m_CommandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, m_pPipeline->fetchGraphicsPipeline());
		m_CommandBuffers[i]->bindVertexBuffers(0, VertexBuffers, Offsets);
		m_CommandBuffers[i]->bindIndexBuffer(vIndexBuffer->fetchBuffer(), 0, vk::IndexType::eUint16);
		m_CommandBuffers[i]->drawIndexed(static_cast<uint32_t>(vIndexBuffer->size() / sizeof(uint16_t)), 1, 0, 0, 0);

		m_CommandBuffers[i]->endRenderPass();
		m_CommandBuffers[i]->end();
	}
}
