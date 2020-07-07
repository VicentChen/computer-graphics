#pragma once
#include "Config.h"
#include "Device.h"
#include "Swapchain.h"
#include "Image.h"

namespace LearnVulkan
{
	class DescriptorPool
	{
	public:
		vk::UniqueDescriptorSetLayout createDescriptorSetLayout(const std::vector<vk::DescriptorSetLayoutBinding>& vBindings = Default::Pipeline::Descriptors);
		std::vector<vk::DescriptorSet>& fetchDescriptorSets() { return m_DescriptorSets; }
		void allocateDescriptorSets(vk::DescriptorSetLayout& vLayout, Image& vImage);
		static DescriptorPool createdByDevice(Device* vDevice) { return vDevice->initDescriptorPool(); }
		
	private:
		DescriptorPool(vk::UniqueDescriptorPool&& vPool, Device* vDevice, Swapchain* vSwapchain) : m_pDevice(vDevice), m_Pool(std::move(vPool)), m_pSwapchain(vSwapchain) { }

		std::vector<vk::DescriptorSet> m_DescriptorSets;
		
		vk::UniqueDescriptorPool m_Pool;
		Device* m_pDevice = nullptr;
		Swapchain* m_pSwapchain = nullptr;

		friend class Device;
	};
}