#pragma once
#include "Config.h"
#include "Device.h"
#include "Shader.h"
#include "Swapchain.h"
#include "RenderPass.h"

namespace LearnVulkan
{
	class Pipeline
	{
	public:
		Pipeline(Device* vDevice, Swapchain* vSwapchain, RenderPass* vRenderPass) : m_pDevice(vDevice), m_pSwapchain(vSwapchain), m_pRenderPass(vRenderPass) {}

		vk::Pipeline fetchGraphicsPipeline() { return m_GraphicsPipeline.get(); }
		vk::PipelineLayout fetchGraphicsPipelineLayout() { return m_GraphicsPipelineLayout.get(); }
		void attachShader(Shader& vShader) { m_Shaders.emplace_back(&vShader); }
		void attachDescriptors(vk::UniqueDescriptorSetLayout& vLayout) { m_Layout = vLayout.get(); }
		void constructGraphicsPipeline();
		
	private:
		std::vector<Shader*> m_Shaders;
		vk::DescriptorSetLayout m_Layout;

		vk::UniquePipeline m_GraphicsPipeline;
		vk::UniquePipelineLayout m_GraphicsPipelineLayout;
		
		Device* m_pDevice = nullptr;
		Swapchain* m_pSwapchain = nullptr;
		RenderPass* m_pRenderPass = nullptr;
	};
}