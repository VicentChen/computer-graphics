#pragma once
#include "Config.h"
#include "Device.h"
#include "Swapchain.h"
#include "Queue.h"
#include "CommandPool.h"
#include "FrameBuffer.h"

namespace LearnVulkan
{
	class Window
	{
	public:
		Window(int vWidth = Default::Window::WIDTH, int vHeight = Default::Window::HEIGHT, const char* vTitle = Default::Window::TITLE.c_str());
		~Window();

		void init(Device* vDevice, Swapchain* vSwapchain, Queue* vGraphicsQueue, Queue* vPresentQueue, FrameBuffer* vBuffer, CommandPool* vCommandPool);
		void display();
		GLFWwindow* getWindowPtr() { return m_pWindow; }
		
	private:
		void __draw(int vCurrentFrame, vk::Semaphore& vRenderSemaphore, vk::Semaphore& vPresentSemaphore, vk::Fence& vFence);
		void __update(int vCurrentFrame);
		
		GLFWwindow* m_pWindow = nullptr;
		Device* m_pDevice = nullptr;
		Swapchain* m_pSwapchain = nullptr;
		Queue* m_pGraphicsQueue = nullptr;
		Queue* m_pPresentQueue = nullptr;
		FrameBuffer* m_pFrameBuffer = nullptr;
		CommandPool* m_pCommandPool = nullptr;
	};
}