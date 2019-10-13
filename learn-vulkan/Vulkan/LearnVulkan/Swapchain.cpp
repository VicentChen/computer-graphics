#include "Swapchain.h"
#include "CommandPool.h"
#include "Queue.h"
#include "Buffer.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
void Swapchain::attachUniformBuffers(CommandPool* vCommandPool, Queue* vGraphicsQueue)
{
	for (int i = 0; i < m_SwapchainImages.size(); i++)
	{
		m_UniformBuffers.emplace_back(m_pDevice->initBuffer(
			vCommandPool,
			vGraphicsQueue,
			nullptr,
			sizeof(Default::Pipeline::UniformTansfromMatrices),
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		));
	}
}

//*********************************************************************
//FUNCTION:
void Swapchain::transfer2UnifromBuffer(const void* vData, uint32_t vSize, int vIndex)
{
	m_pDevice->transferData(vData, vSize, m_UniformBuffers[vIndex]);
}
