#pragma once
#include "Config.h"
#include "Device.h"
#include "Queue.h"
#include "Swapchain.h"
#include "RenderPass.h"
#include "FrameBuffer.h"
#include "Pipeline.h"
#include "Buffer.h"
#include "DescriptorPool.h"

namespace LearnVulkan
{
	class Window;
	class CommandPool
	{
	public:
		CommandPool(Device* vDevice,
			Swapchain* vSwapchain,
			Queue* vGraphicsQueue,
			RenderPass* vRenderPass,
			FrameBuffer* vFrameBuffer,
			Pipeline* vPipeline,
			DescriptorPool* vDescriptorPool)
		: m_pDevice(vDevice),
		m_pSwapchain(vSwapchain),
		m_pGraphicsQueue(vGraphicsQueue),
		m_pRenderPass(vRenderPass),
		m_pFrameBuffer(vFrameBuffer),
		m_pPipeline(vPipeline),
		m_pDescriptorPool(vDescriptorPool){}
		
		void constructCommandPool();
		void constructCommandBuffers(std::vector<Buffer*>& vVertexBuffers, Buffer* vIndexBuffer);
		vk::CommandPool fetchCommandPool() { return m_CommandPool.get(); }
		vk::CommandBuffer& fetchCommandBufferAt(int vIndex) { return m_CommandBuffers[vIndex].get(); }
		
	private:

		vk::UniqueCommandPool m_CommandPool;
		std::vector<vk::UniqueCommandBuffer> m_CommandBuffers;
		
		Device* m_pDevice = nullptr;
		Queue* m_pGraphicsQueue = nullptr;
		Swapchain* m_pSwapchain = nullptr;
		RenderPass* m_pRenderPass = nullptr;
		FrameBuffer* m_pFrameBuffer = nullptr;
		Pipeline* m_pPipeline = nullptr;
		DescriptorPool* m_pDescriptorPool = nullptr;

		friend class Window;
	};
}
