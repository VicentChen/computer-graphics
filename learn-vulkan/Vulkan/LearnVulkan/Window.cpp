#include "Window.h"
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>

using namespace LearnVulkan;

Window::Window(int vWidth, int vHeight, const char* vTitle)
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	m_pWindow = glfwCreateWindow(vWidth, vHeight, vTitle, nullptr, nullptr);

	glfwSetWindowUserPointer(m_pWindow, this);
	glfwSetFramebufferSizeCallback(m_pWindow, windowResizeCallback);
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
	std::vector<vk::UniqueSemaphore> Ready2RenderSemaphores ;
	std::vector<vk::UniqueSemaphore> Ready2PresentSemaphores;
	std::vector<vk::UniqueFence> InFlightFences;

	for (size_t i = 0; i < Default::Window::FRAMES_IN_FLIGHT; i++)
	{
		Ready2RenderSemaphores.emplace_back(m_pDevice->fetchDevice().createSemaphoreUnique(Default::Device::SemaphoreInfo));
		Ready2PresentSemaphores.emplace_back(m_pDevice->fetchDevice().createSemaphoreUnique(Default::Device::SemaphoreInfo));
		InFlightFences.emplace_back(m_pDevice->fetchDevice().createFenceUnique(Default::Device::FenceInfo));
	}
	
	for(int CurrentFrame = 0; !glfwWindowShouldClose(m_pWindow); CurrentFrame = (CurrentFrame + 1) % Default::Window::FRAMES_IN_FLIGHT)
	{
		glfwPollEvents();
		
		vk::Semaphore& RenderSemaphore = Ready2RenderSemaphores[CurrentFrame].get();
		vk::Semaphore& PresentSemaphore = Ready2PresentSemaphores[CurrentFrame].get();
		vk::Fence& InFlightFence = InFlightFences[CurrentFrame].get();

		m_pDevice->fetchDevice().waitForFences(1, &InFlightFence, true, UINT64_MAX);
		m_pDevice->fetchDevice().resetFences(1, &InFlightFence);
		__draw(CurrentFrame, RenderSemaphore, PresentSemaphore, InFlightFence);
	}
	m_pDevice->fetchDevice().waitIdle();
}

//*********************************************************************
//FUNCTION:
void Window::__draw(int vCurrentFrame, vk::Semaphore& vRenderSemaphore, vk::Semaphore& vPresentSemaphore, vk::Fence& vFence)
{
	uint32_t ImageIndex;
	vkAcquireNextImageKHR(m_pDevice->fetchDevice(), m_pSwapchain->fetchSwapchain().get(), UINT64_MAX, vRenderSemaphore, VK_NULL_HANDLE, &ImageIndex);
	
	vk::PipelineStageFlags Stages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	vk::CommandBuffer& Buffer = m_pCommandPool->fetchCommandBufferAt(ImageIndex);
	__update(vCurrentFrame);
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
	vkQueueSubmit(m_pGraphicsQueue->fetchQueue(), 1, &RealSubmitInfo, vFence);

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

//*********************************************************************
//FUNCTION:
void Window::__update(int vCurrentFrame)
{
	static auto StartTime = std::chrono::high_resolution_clock::now();
	auto CurrentTime = std::chrono::high_resolution_clock::now();
	float Time = std::chrono::duration<float, std::chrono::seconds::period>(CurrentTime - StartTime).count();

	Default::Pipeline::UniformTansfromMatrices T;
	T.Model = glm::rotate(glm::mat4(1.0f), Time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));;
	T.View  = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	T.Proj  = glm::perspective(glm::radians(45.0f), m_pSwapchain->getExtent().width / (float)m_pSwapchain->getExtent().height, 0.1f, 10.0f);
	T.Proj[1][1] *= -1;
	
	m_pSwapchain->transfer2UnifromBuffer(&T, sizeof(T), vCurrentFrame);
}
