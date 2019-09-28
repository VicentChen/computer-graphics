#pragma once
#include "Config.h"
#include "Device.h"

namespace LearnVulkan
{
	class Shader
	{
	public:
		Shader(vk::UniqueShaderModule&& vShader,
			const std::vector<vk::VertexInputBindingDescription>& vBindings,
			const std::vector<vk::VertexInputAttributeDescription>& vAttributes,
			const vk::PipelineShaderStageCreateInfo& vStageInfo)
		: m_Shader(std::move(vShader)),
		m_Bindings(vBindings),
		m_Attributes(vAttributes),
		m_StageInfo(vStageInfo) {}

		const vk::PipelineShaderStageCreateInfo& getStageInfo() const { return m_StageInfo; }
		const std::vector<vk::VertexInputBindingDescription> getBindings() const { return m_Bindings; }
		const std::vector<vk::VertexInputAttributeDescription> getAttributes() const { return m_Attributes; }
		
		static Shader createdByDevice(Device& vDevice,
			vk::ShaderStageFlagBits vStage,
			const std::string& vShaderFilePath,
			std::vector<vk::VertexInputBindingDescription>& vBindings,
			std::vector<vk::VertexInputAttributeDescription>& vAttributes,
			const std::string& vEntrance = Default::Shader::Entrance)
		{
			return vDevice.initShader(vStage, vShaderFilePath, vBindings, vAttributes, vEntrance);
		}
	private:
		vk::PipelineShaderStageCreateInfo m_StageInfo;
		
		std::vector<vk::VertexInputBindingDescription> m_Bindings;
		std::vector<vk::VertexInputAttributeDescription> m_Attributes;
		
		vk::UniqueShaderModule m_Shader;
	};
}