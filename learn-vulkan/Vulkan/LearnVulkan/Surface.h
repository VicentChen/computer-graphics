#pragma once
#include "Config.h"
#include "Instance.h"

namespace LearnVulkan
{
	class Surface
	{
	public:
		vk::UniqueSurfaceKHR& fetchSurface() { return m_Surface; }

		static Surface createByInstanceWithGLFW(GLFWwindow* vWindow, Instance& vInstance);
		
	private:
		Surface(vk::UniqueSurfaceKHR&& vSurface) : m_Surface(std::move(vSurface)) {}
		
		vk::UniqueSurfaceKHR m_Surface;
	};
}
