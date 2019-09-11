#include "Window.h"

using namespace LearnVulkan;

Window::Window(int vWidth, int vHeight, const char* vTitle)
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	m_pWindow = glfwCreateWindow(vWidth, vHeight, vTitle, nullptr, nullptr);
}

Window::~Window()
{
	glfwDestroyWindow(m_pWindow);
	glfwTerminate();
}

void Window::display()
{
	while(!glfwWindowShouldClose(m_pWindow))
	{
		glfwPollEvents();
	}
}
