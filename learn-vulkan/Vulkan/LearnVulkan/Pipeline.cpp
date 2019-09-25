#include "Pipeline.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
void Pipeline::constructGraphicsPipeline()
{
	std::vector<vk::PipelineShaderStageCreateInfo> ShaderStages;
	for (const auto& pShader : m_Shaders)
		ShaderStages.emplace_back(pShader->getStageInfo());
	
	std::vector<vk::VertexInputBindingDescription> Bindings;
	std::vector<vk::VertexInputAttributeDescription> Attributes;

	for (const auto& pShader : m_Shaders)
	{
		auto Binding = pShader->getBindings();
		auto Attribute = pShader->getAttributes();
		Bindings.insert(Bindings.end(), Binding.begin(), Binding.end());
		Attributes.insert(Attributes.end(), Attribute.begin(), Attribute.end());
	}
	
	vk::PipelineVertexInputStateCreateInfo VertexInputStateCreateInfo = {
		vk::PipelineVertexInputStateCreateFlags(),
		CAST_U32I(Bindings.size()),
		Bindings.data(),
		CAST_U32I(Attributes.size()),
		Attributes.data()
	};

	vk::PipelineInputAssemblyStateCreateInfo InputAssemblyStateCreateInfo = {
		vk::PipelineInputAssemblyStateCreateFlags(),
		vk::PrimitiveTopology::eTriangleList,
		false
	};

	vk::Extent2D SwapchainExtent = m_pSwapchain->getExtent();
	vk::Viewport Viewport = {
		0, 0,
		CAST_FLT(SwapchainExtent.width), CAST_FLT(SwapchainExtent.height),
		0, 1
	};
	vk::Rect2D Scissors = { {0, 0}, SwapchainExtent };
	vk::PipelineViewportStateCreateInfo ViewportStateCreateInfo = {
		vk::PipelineViewportStateCreateFlags(),
		1,
		&Viewport,
		1,
		&Scissors
	};

	vk::PipelineRasterizationStateCreateInfo RasterizationStateCreateInfo = {
		vk::PipelineRasterizationStateCreateFlags(),
		false,
		false,
		vk::PolygonMode::eFill,
		vk::CullModeFlagBits::eBack,
		vk::FrontFace::eClockwise,
		false,
		0,
		0,
		0,
		1.0
	};

	vk::PipelineMultisampleStateCreateInfo MultisampleStateCreateInfo = {
		vk::PipelineMultisampleStateCreateFlags(),
		vk::SampleCountFlagBits::e1,
		false,
		1.0f,
		nullptr,
		false,
		false
	};

	vk::PipelineColorBlendAttachmentState ColorBlendAttachmentState = {
		false,
		vk::BlendFactor::eOne,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::BlendFactor::eOne,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};
	vk::PipelineColorBlendStateCreateInfo ColorBlendCreateInfo = {
		vk::PipelineColorBlendStateCreateFlags(),
		false,
		vk::LogicOp::eCopy,
		1,
		&ColorBlendAttachmentState
	};

	std::vector<vk::DynamicState> DynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eLineWidth };
	vk::PipelineDynamicStateCreateInfo DynamicStateCreateInfo = {
		vk::PipelineDynamicStateCreateFlags(),
		CAST_U32I(DynamicStates.size()),
		DynamicStates.data()
	};

	vk::PipelineLayoutCreateInfo LayoutCreateInfo = {};
	vk::UniquePipelineLayout Layout = m_pDevice->fetchDevice().createPipelineLayoutUnique(LayoutCreateInfo);

	vk::GraphicsPipelineCreateInfo Info = {
		vk::PipelineCreateFlags(),
		CAST_U32I(ShaderStages.size()),
		ShaderStages.data(),
		&VertexInputStateCreateInfo,
		&InputAssemblyStateCreateInfo,
		nullptr,
		&ViewportStateCreateInfo,
		&RasterizationStateCreateInfo,
		&MultisampleStateCreateInfo,
		nullptr,
		&ColorBlendCreateInfo,
		nullptr,
		Layout.get(),
		m_pRenderPass->fetchRenderPass()
	};

	m_GraphicsPipeline = m_pDevice->fetchDevice().createGraphicsPipelineUnique(nullptr, Info);
}
