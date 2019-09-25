#pragma once
#include "Config.h"
#include "Swapchain.h"
#include "RenderPass.h"

namespace LearnVulkan
{
	class FrameBuffer
	{
	public:
		FrameBuffer(Device* vDevice, Swapchain* vSwapchain, RenderPass* vRenderPass) : m_pDevice(vDevice), m_pSwapchain(vSwapchain), m_pRenderPass(vRenderPass) {}
		vk::Framebuffer& fetchFrameBufferAt(int i) { return m_FrameBuffers[i].get(); }
		void constructFrameBuffer();
		
	private:
		Device* m_pDevice = nullptr;
		Swapchain* m_pSwapchain = nullptr;
		RenderPass* m_pRenderPass = nullptr;
		
		std::vector<vk::UniqueFramebuffer> m_FrameBuffers;
	};
}