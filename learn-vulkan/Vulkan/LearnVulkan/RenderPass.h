#pragma once
#include "Config.h"
#include "Device.h"
#include "Swapchain.h"

namespace LearnVulkan
{
	class RenderPass
	{
	public:
		RenderPass(Device* vDevice, Swapchain* vSwapchain) : m_pDevice(vDevice), m_pSwapchain(vSwapchain) {}

		vk::RenderPass& fetchRenderPass() { return m_RenderPass.get(); }
		
		void constructRenderPass();

	private:
		vk::UniqueRenderPass m_RenderPass;
		
		Device* m_pDevice = nullptr;
		Swapchain* m_pSwapchain = nullptr;
	};
}