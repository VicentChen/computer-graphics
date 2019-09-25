#include "Surface.h"

//*********************************************************************
//FUNCTION:
LearnVulkan::Surface LearnVulkan::Surface::createByInstanceWithGLFW(GLFWwindow* vWindow, Instance& vInstance)
{
	VkSurfaceKHR VkSurface;
	if (glfwCreateWindowSurface(vInstance.fetchInstance(), vWindow, nullptr, &VkSurface) != VK_SUCCESS)
		VERBOSE_EXIT("Failed to create surface");
	vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderStatic> SurfaceDeleter(vInstance.fetchInstance());
	return Surface(vk::UniqueSurfaceKHR(std::move(VkSurface), SurfaceDeleter));
}
