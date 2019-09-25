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

//*********************************************************************
//FUNCTION:
void Window::init(Device* vDevice, Swapchain* vSwapchain, Queue* vGraphicsQueue, Queue* vPresentQueue, FrameBuffer* vFrameBuffer, CommandPool* vCommandPool)
{
	m_pDevice = vDevice;
	m_pSwapchain = vSwapchain;
	m_pFrameBuffer = vFrameBuffer;
	m_pCommandPool = vCommandPool;
	m_pGraphicsQueue = vGraphicsQueue;
	m_pPresentQueue = vPresentQueue;
}

//*********************************************************************
//FUNCTION:
void Window::display()
{
	vk::UniqueSemaphore Ready2RenderSemaphore = m_pDevice->fetchDevice().createSemaphoreUnique(Default::Device::SemaphoreInfo);
	vk::UniqueSemaphore Ready2PresentSemaphore = m_pDevice->fetchDevice().createSemaphoreUnique(Default::Device::SemaphoreInfo);
	vk::Semaphore& RenderSemaphore = Ready2RenderSemaphore.get();
	vk::Semaphore& PresentSemaphore = Ready2PresentSemaphore.get();
	while(!glfwWindowShouldClose(m_pWindow))
	{
		glfwPollEvents();
		__draw(RenderSemaphore, PresentSemaphore);
	}
	m_pDevice->fetchDevice().waitIdle();
}

//*********************************************************************
//FUNCTION:
void Window::__draw(vk::Semaphore& vRenderSemaphore, vk::Semaphore& vPresentSemaphore)
{

	uint32_t ImageIndex;
	vkAcquireNextImageKHR(m_pDevice->fetchDevice(), m_pSwapchain->fetchSwapchain().get(), UINT64_MAX, vRenderSemaphore, VK_NULL_HANDLE, &ImageIndex);
	
	vk::PipelineStageFlags Stages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	vk::CommandBuffer& Buffer = m_pCommandPool->fetchCommandBufferAt(ImageIndex);
	vk::SubmitInfo SubmitInfo = {
		1,
		&vRenderSemaphore,
		&Stages,
		1,
		&Buffer,
		1,
		&vPresentSemaphore
	};
	VkSubmitInfo RealSubmitInfo = SubmitInfo;
	vkQueueSubmit(m_pGraphicsQueue->fetchQueue(), 1, &RealSubmitInfo, VK_NULL_HANDLE);

	auto& Swapchain = m_pSwapchain->fetchSwapchain().get();
	vk::PresentInfoKHR PresentInfo = {
		1,
		&vPresentSemaphore,
		1,
		&Swapchain,
		&ImageIndex
	};
	m_pPresentQueue->fetchQueue().presentKHR(PresentInfo);
}
