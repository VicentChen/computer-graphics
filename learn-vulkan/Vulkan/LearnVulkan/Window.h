#pragma once
#include "Common.h"
#include "Config.h"
#include <GLFW/glfw3.h>

namespace LearnVulkan
{
	class Window
	{
	public:
		Window(int vWidth = Default::Window::WIDTH, int vHeight = Default::Window::HEIGHT, const char* vTitle = Default::Window::TITLE.c_str());
		~Window();

		void display();
		
	private:
		GLFWwindow* m_pWindow = nullptr;
	};
}