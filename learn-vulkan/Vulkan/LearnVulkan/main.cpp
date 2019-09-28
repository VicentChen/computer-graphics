#include "Config.h"
#include "Instance.h"
#include "Device.h"
#include "Surface.h"
#include "Window.h"
#include "Queue.h"
#include "Swapchain.h"
#include "Pipeline.h"
#include "RenderPass.h"
#include "FrameBuffer.h"
#include "CommandPool.h"
#include "Buffer.h"

using namespace LearnVulkan;

int main(int argc, char* argv[])
{
	Window window;
	Instance instance(Default::Application::Info);
	Surface surface = Surface::createByInstanceWithGLFW(window.getWindowPtr(), instance);
	
	PhysicalDevice physical_device = instance.initPhysicalDevice(surface);
	Device device = physical_device.initDevice();
	Queue graphics_queue = device.initQueue(Default::PhysicalDevice::GRAPHICS_QUEUE_NAME);
	Queue& present_queue = graphics_queue;
	Swapchain swapchain = device.initSwapchain(physical_device, surface);
	
	Shader vert_shader = device.initShader(vk::ShaderStageFlagBits::eVertex, Default::Shader::VertexPath, Default::Shader::VertexInputBindings, Default::Shader::VertexInputAttributes, Default::Shader::Entrance);
	Shader frag_shader = device.initShader(vk::ShaderStageFlagBits::eFragment, Default::Shader::FragmentPath, {}, {}, Default::Shader::Entrance);
	RenderPass render_pass(&device, &swapchain);
	render_pass.constructRenderPass();
	FrameBuffer framebuffer(&device, &swapchain, &render_pass);
	framebuffer.constructFrameBuffer();
	
	Pipeline pipeline(&device, &swapchain, &render_pass);
	pipeline.attachShader(vert_shader);
	pipeline.attachShader(frag_shader);
	pipeline.constructGraphicsPipeline();

	Buffer coord_buffer = device.initVertexBuffer(Default::Shader::CoordBufferInfo, Default::Shader::Vertices.data(), Default::Shader::Vertices.size() * sizeof(glm::vec3));
	Buffer color_buffer = device.initVertexBuffer(Default::Shader::ColorBufferInfo, Default::Shader::Colors.data(), Default::Shader::Colors.size() * sizeof(glm::vec3));
	std::vector<Buffer*> buffers = { &coord_buffer, &color_buffer };
	
	CommandPool command_pool(&device, &swapchain, &graphics_queue, &render_pass, &framebuffer, &pipeline, buffers);
	command_pool.constructCommandPool();
	command_pool.constructCommandBuffers();
	
	window.init(&device, &swapchain, &graphics_queue, &present_queue, &framebuffer, &command_pool);
	window.display();
	return 0;
}