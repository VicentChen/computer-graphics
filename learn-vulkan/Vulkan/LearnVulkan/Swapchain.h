#pragma once
#include "Device.h"
#include "Surface.h"

namespace LearnVulkan
{
	class Swapchain
	{
	public:

		vk::UniqueSwapchainKHR& fetchSwapchain() { return m_Swapchain; }
		std::vector<vk::UniqueImageView>& fetchImageViews() { return m_SwapchainImageViews; }
		
		const vk::SurfaceFormatKHR getFormat() const { return m_Format; }
		const vk::Extent2D getExtent() const { return m_Extent; }
		
		static Swapchain createByDevice(Device& vDevice, PhysicalDevice& vPhysicalDevice, Surface& vSurface, uint32_t vWidth = Default::Window::WIDTH, uint32_t vHeight = Default::Window::HEIGHT)
		{
			return vDevice.initSwapchain(vPhysicalDevice, vSurface, vWidth, vHeight);
		}
		
	private:
		Swapchain(vk::UniqueSwapchainKHR&& vSwapchain, Surface* vSurface, PhysicalDevice* vPhysicalDevice, Device* vDevice) :
		m_Swapchain(std::move(vSwapchain)),
		m_pSurface(vSurface),
		m_pPhysicalDevice(vPhysicalDevice),
		m_pDevice(vDevice) { }
		
		vk::UniqueSwapchainKHR m_Swapchain;
		std::vector<vk::Image> m_SwapchainImages;
		std::vector<vk::UniqueImageView> m_SwapchainImageViews;

		vk::Extent2D m_Extent;
		vk::SurfaceFormatKHR m_Format;
		
		// hack
		Surface* m_pSurface = nullptr;
		PhysicalDevice* m_pPhysicalDevice = nullptr;
		Device* m_pDevice = nullptr;

		friend class Device;
	};
}