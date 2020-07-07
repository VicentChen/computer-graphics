#include "DescriptorPool.h"
#include "Buffer.h"

using namespace LearnVulkan;

//*********************************************************************
//FUNCTION:
vk::UniqueDescriptorSetLayout DescriptorPool::createDescriptorSetLayout(const std::vector<vk::DescriptorSetLayoutBinding>& vBindings)
{
	vk::DescriptorSetLayoutCreateInfo Info = {
		vk::DescriptorSetLayoutCreateFlags(),
		CAST_U32I(vBindings.size()),
		vBindings.data()
	};
	return m_pDevice->fetchDevice().createDescriptorSetLayoutUnique(Info);
}

//*********************************************************************
//FUNCTION:
void DescriptorPool::allocateDescriptorSets(vk::DescriptorSetLayout& vLayout, Image& vImage)
{
	std::vector<vk::DescriptorSetLayout> Layouts;
	Layouts.resize(m_pSwapchain->fetchImageViews().size(), vLayout);
	
	vk::DescriptorSetAllocateInfo AllocateInfo = {
		m_Pool.get(),
		CAST_U32I(Layouts.size()),
		Layouts.data()
	};
	m_DescriptorSets = m_pDevice->fetchDevice().allocateDescriptorSets(AllocateInfo);

	auto& UniformBuffers = m_pSwapchain->fetchUniformBuffers();

	for (int i = 0; i < m_DescriptorSets.size(); i++)
	{
		vk::DescriptorBufferInfo BufferInfo = {
			UniformBuffers[i].fetchBuffer(),
			0,
			sizeof(Default::Pipeline::UniformTansfromMatrices)
		};

		vk::DescriptorImageInfo ImageInfo = {
			vImage.fetchSampler(),
			vImage.fetchImageView(),
			vk::ImageLayout::eShaderReadOnlyOptimal
		};
		
		std::vector<vk::WriteDescriptorSet> DescriptorWrites;
		DescriptorWrites.emplace_back(vk::WriteDescriptorSet{ 
			m_DescriptorSets[i],
			0,
			0,
			1,
			vk::DescriptorType::eUniformBuffer,
			nullptr,
			&BufferInfo });
		DescriptorWrites.emplace_back(vk::WriteDescriptorSet{
			m_DescriptorSets[i],
			1,
			0,
			1,
			vk::DescriptorType::eCombinedImageSampler,
			&ImageInfo });

		m_pDevice->fetchDevice().updateDescriptorSets(DescriptorWrites, nullptr);
	}
}

