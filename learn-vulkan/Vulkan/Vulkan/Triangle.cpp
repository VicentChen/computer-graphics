#include "Triangle.h"
#include <algorithm>
#include <set>
#include <string>

VkResult CreateDebugUtilsMessengerEXT(
	VkInstance vInstance, 
	const VkDebugUtilsMessengerCreateInfoEXT* vCreateInfo, 
	const VkAllocationCallbacks* vAllocator, 
	VkDebugUtilsMessengerEXT* vDebugMessenger)
{
	auto Func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vInstance, "vkCreateDebugUtilsMessengerEXT");
	return Func != nullptr ? Func(vInstance, vCreateInfo, vAllocator, vDebugMessenger) : VK_ERROR_EXTENSION_NOT_PRESENT;
}

VkResult DestroyDebugUtilsMessengerEXT(
	VkInstance vInstance, 
	VkDebugUtilsMessengerEXT vDebugMessenger, 
	const VkAllocationCallbacks* vAllocator)
{
	auto Func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vInstance, "vkDestroyDebugUtilsMessengerEXT");
	if (Func != nullptr) Func(vInstance, vDebugMessenger, vAllocator);
	return VK_SUCCESS;
}

VkBool32 HelloTriangleApplication::debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT vMessageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT vMessageType,
	const VkDebugUtilsMessengerCallbackDataEXT* vCallbackData,
	void* vUserData)
{
	std::cerr << "[Validation Layer] " << vCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

std::vector<char> HelloTriangleApplication::readFile(const std::string& vFileName)
{
	std::ifstream File(vFileName, std::ios::ate | std::ios::binary);
	if (!File.is_open()) VERBOSE_EXIT("failed to open file");

	size_t FileSize = static_cast<size_t>(File.tellg());
	std::vector<char> Buffer(FileSize);
	File.seekg(0);
	File.read(Buffer.data(), FileSize);
	File.close();
	return Buffer;
}

void HelloTriangleApplication::run()
{
	initWindow();
	initVulkan();
	mainLoop();
	cleanup();
}

void HelloTriangleApplication::initWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	m_pWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan", nullptr, nullptr);
}

void HelloTriangleApplication::initVulkan()
{
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createRenderPass();
	createGraphicsPipeline();
	createFrameBuffers();
	createCommandPool();
	createCommandBuffers();
	createSyncObjects();
}

void HelloTriangleApplication::createInstance()
{
	if (EnableValidationLayers && !checkValidationLayerSupport())
		VERBOSE_EXIT("validation layers requested, but not available!");
	
	VkApplicationInfo AppInfo = {};
	AppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	AppInfo.pApplicationName = "Triangle";
	AppInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	AppInfo.pEngineName = "No Engine";
	AppInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	AppInfo.apiVersion = VK_API_VERSION_1_0;

	std::vector<const char*> RequiredExtensions = getRequiredExtensions();
	VkInstanceCreateInfo CreateInfo = {};
	CreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	CreateInfo.pApplicationInfo = &AppInfo;
	CreateInfo.enabledExtensionCount = static_cast<uint32_t>(RequiredExtensions.size());
	CreateInfo.ppEnabledExtensionNames = RequiredExtensions.data();
	
	VkDebugUtilsMessengerCreateInfoEXT DebugCreateInfo;
	if constexpr (EnableValidationLayers)
	{
		CreateInfo.enabledLayerCount = static_cast<uint32_t>(ValidationLayers.size());
		CreateInfo.ppEnabledLayerNames = ValidationLayers.data();
		populateDebugMessengerCreateInfo(DebugCreateInfo);
		CreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&DebugCreateInfo;
	}
	else
	{
		CreateInfo.enabledLayerCount = 0;
		CreateInfo.pNext = nullptr;
	}

#if CREATE_INSTANCE_VERBOSE
	OUTPUT_HEADER();
	uint32_t SupportedExtensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &SupportedExtensionCount, nullptr);
	std::vector<VkExtensionProperties> SupportedExtensions(SupportedExtensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &SupportedExtensionCount, SupportedExtensions.data());

	std::cout << "Supported Extensions: " << std::endl;
	for (const auto& Extension : SupportedExtensions) std::cout << "\t" << Extension.extensionName << std::endl;
	std::cout << "Required Extensions: " << std::endl;
	for (const char* pRequiredExtension : RequiredExtensions)
	{
		const VkExtensionProperties* pExtension = reinterpret_cast<const VkExtensionProperties*>(pRequiredExtension);
		std::cout << '\t' << pExtension->extensionName << std::endl;
	}
#endif

	// ----- create instance ----- //
	if (!IS_VK_SUCCESS(vkCreateInstance(&CreateInfo, nullptr, &m_Instance)))
		VERBOSE_EXIT("failed to create instance");
}

bool HelloTriangleApplication::checkValidationLayerSupport()
{
	uint32_t LayerCount = 0;
	vkEnumerateInstanceLayerProperties(&LayerCount, nullptr);
	std::vector<VkLayerProperties> AvailableLayers(LayerCount);
	vkEnumerateInstanceLayerProperties(&LayerCount, AvailableLayers.data());

#if CHECK_VALIDATION_LAYER_SUPPORT_VERBOSE
	OUTPUT_HEADER();
	std::cout << "Layers requested: " << std::endl;
	for (const char* pLayerName : ValidationLayers) std::cout << '\t' << pLayerName << std::endl;
	std::cout << "Layers support: " << std::endl;
	for (const VkLayerProperties& Layer : AvailableLayers) std::cout << '\t' << Layer.layerName << std::endl;
#endif
	
	for (const char* pLayerName : ValidationLayers)
	{
		bool LayerFound = false;
		for (const VkLayerProperties& Layer : AvailableLayers)
		{
			if (std::strcmp(pLayerName, Layer.layerName) == 0)
			{
				LayerFound = true;
				break;
			}
		}
		if (!LayerFound) return false;
	}
	return true;
}

std::vector<const char*> HelloTriangleApplication::getRequiredExtensions()
{
	uint32_t GLFWExtensionCount = 0;
	const char** pGLFWExtension = glfwGetRequiredInstanceExtensions(&GLFWExtensionCount);
	std::vector<const char*> Extensions(pGLFWExtension, pGLFWExtension + GLFWExtensionCount);
	if constexpr (EnableValidationLayers) Extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	return Extensions;
}

void HelloTriangleApplication::setupDebugMessenger()
{
	if constexpr (!EnableValidationLayers) return;
	VkDebugUtilsMessengerCreateInfoEXT CreateInfo;
	populateDebugMessengerCreateInfo(CreateInfo);
	if (!IS_VK_SUCCESS(CreateDebugUtilsMessengerEXT(m_Instance, &CreateInfo, nullptr, &m_DebugMessenger)))
		VERBOSE_EXIT("failed to setup debug messenger");
}

void HelloTriangleApplication::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& voCreateInfo)
{
	voCreateInfo = {};
	voCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	voCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	voCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	voCreateInfo.pfnUserCallback = debugCallback;
	voCreateInfo.pUserData = nullptr;
}

void HelloTriangleApplication::pickPhysicalDevice()
{
	uint32_t DeviceCount = 0;
	vkEnumeratePhysicalDevices(m_Instance, &DeviceCount, nullptr);
	if (DeviceCount == 0) VERBOSE_EXIT("failed to find GPUS with Vulkan support");

	std::vector<VkPhysicalDevice> Devices(DeviceCount);
	vkEnumeratePhysicalDevices(m_Instance, &DeviceCount, Devices.data());

#if PICK_PHYSICAL_DEVICE_VERBOSE
	OUTPUT_HEADER();
	std::cout << DeviceCount << " Device founded:" << std::endl;
	for(const auto& Device : Devices)
	{
		VkPhysicalDeviceProperties DeviceProperties;
		vkGetPhysicalDeviceProperties(Device, &DeviceProperties);
		std::cout << "\tDevice " << DeviceProperties.deviceID << " - " << DeviceProperties.deviceName << std::endl;
	}
#endif

	for (const auto& Device : Devices)
	{
		if (isDeviceSuitable(Device))
		{
			m_PhysicalDevice = Device;
			break;
		}
	}
	
	if (m_PhysicalDevice == VK_NULL_HANDLE) VERBOSE_EXIT("failed to find a suitable device");
}

bool HelloTriangleApplication::isDeviceSuitable(VkPhysicalDevice vDevice)
{
	QueueFamilyIndices Indices = findQueueFamilies(vDevice);
	bool ExtensionSupported = checkDeviceExtensionSupport(vDevice);

	bool SwapChainAdequate = false;
	if (ExtensionSupported)
	{
		SwapChainSupportDetails SwapChainSupport = querySwapChainSupport(vDevice);
		SwapChainAdequate = !SwapChainSupport.Formats.empty() && !SwapChainSupport.PresentModes.empty();
	}
	return Indices.isComplete() && ExtensionSupported && SwapChainAdequate;
}

SwapChainSupportDetails HelloTriangleApplication::querySwapChainSupport(VkPhysicalDevice vDevice)
{
	SwapChainSupportDetails Details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vDevice, m_Surface, &Details.Capabilities);

	uint32_t FormatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(vDevice, m_Surface, &FormatCount, nullptr);
	if (FormatCount != 0)
	{
		Details.Formats.resize(FormatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(vDevice, m_Surface, &FormatCount, Details.Formats.data());
	}

	uint32_t PresentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(vDevice, m_Surface, &PresentModeCount, nullptr);
	if (PresentModeCount != 0)
	{
		Details.PresentModes.resize(PresentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(vDevice, m_Surface, &PresentModeCount, Details.PresentModes.data());
	}
	
	return Details;
}

QueueFamilyIndices HelloTriangleApplication::findQueueFamilies(VkPhysicalDevice vDevice)
{
	QueueFamilyIndices Indices;

	uint32_t QueueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(vDevice, &QueueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> QueueFamilies(QueueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(vDevice, &QueueFamilyCount, QueueFamilies.data());

	for (int i = 0; i < QueueFamilies.size(); i++)
	{
		if (Indices.isComplete()) break;
		if (QueueFamilies[i].queueCount > 0 && QueueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			Indices.GraphicsFamily = i;

		VkBool32 PresentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(vDevice, i, m_Surface, &PresentSupport);
		if (QueueFamilies[i].queueCount > 0 && PresentSupport)
			Indices.PresentFamily = i;
	}
	return Indices;
}

bool HelloTriangleApplication::checkDeviceExtensionSupport(VkPhysicalDevice vDevice)
{
	uint32_t ExtensionCount = 0;
	vkEnumerateDeviceExtensionProperties(vDevice, nullptr, &ExtensionCount, nullptr);
	std::vector<VkExtensionProperties> SupportedExtensions(ExtensionCount);
	vkEnumerateDeviceExtensionProperties(vDevice, nullptr, &ExtensionCount, SupportedExtensions.data());
	std::set<std::string> RequestedExtensions(DeviceExtensions.begin(), DeviceExtensions.end());

#if CHECK_DEVICE_EXTENSION_SUPPORT_VERBOSE
	OUTPUT_HEADER();
	std::cout << "Supported extensions: " << std::endl;
	for (const auto& Extension : SupportedExtensions) std::cout << '\t' << Extension.extensionName << std::endl;
	std::cout << "Requested extensions: " << std::endl;
	for (const auto& Extension : RequestedExtensions) std::cout << '\t' << Extension << std::endl;
#endif
	
	for (const auto& Extension : SupportedExtensions)
		RequestedExtensions.erase(Extension.extensionName);
	return RequestedExtensions.empty();
}

void HelloTriangleApplication::createLogicalDevice()
{
	QueueFamilyIndices Indices = findQueueFamilies(m_PhysicalDevice);

	std::vector<VkDeviceQueueCreateInfo> QueueCreateInfos;
	std::set<uint32_t> UniqueQueueFamilies = {
		Indices.GraphicsFamily.value(),
		Indices.PresentFamily.value()
	};
	
	float QueuePriority = 1.0f;
	for (uint32_t QueueFamily : UniqueQueueFamilies)
	{
		VkDeviceQueueCreateInfo QueueCreateInfo = {};
		QueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		QueueCreateInfo.queueFamilyIndex = QueueFamily;
		QueueCreateInfo.queueCount = 1;
		QueueCreateInfo.pQueuePriorities = &QueuePriority;
		QueueCreateInfos.push_back(QueueCreateInfo);
	}

	VkPhysicalDeviceFeatures DeviceFeatures = {};
	VkDeviceCreateInfo DeviceCreateInfo = {};
	DeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	DeviceCreateInfo.pQueueCreateInfos = QueueCreateInfos.data();
	DeviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(QueueCreateInfos.size());
	DeviceCreateInfo.pEnabledFeatures = &DeviceFeatures;
	DeviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(DeviceExtensions.size());
	DeviceCreateInfo.ppEnabledExtensionNames = DeviceExtensions.data();
	if constexpr (EnableValidationLayers)
	{
		DeviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(ValidationLayers.size());
		DeviceCreateInfo.ppEnabledLayerNames = ValidationLayers.data();
	}
	else
	{
		DeviceCreateInfo.enabledLayerCount = 0;
	}

	if (!IS_VK_SUCCESS(vkCreateDevice(m_PhysicalDevice, &DeviceCreateInfo, nullptr, &m_Device)))
		VERBOSE_EXIT("failed to create logical device");

	vkGetDeviceQueue(m_Device, Indices.GraphicsFamily.value(), 0, &m_GraphicsQueue);
	vkGetDeviceQueue(m_Device, Indices.PresentFamily.value(), 0, &m_PresentQueue);
}

void HelloTriangleApplication::createSurface()
{
	if (!IS_VK_SUCCESS(glfwCreateWindowSurface(m_Instance, m_pWindow, nullptr, &m_Surface)))
		VERBOSE_EXIT("failed to create window surface");
}

void HelloTriangleApplication::createSwapChain()
{
	SwapChainSupportDetails SwapChainSupport = querySwapChainSupport(m_PhysicalDevice);
	VkSurfaceFormatKHR SurfaceFormat = chooseSwapSurfaceFormat(SwapChainSupport.Formats);
	VkPresentModeKHR PresentMode = chooseSwapPresentMode(SwapChainSupport.PresentModes);
	VkExtent2D Extent = chooseSwapExtent(SwapChainSupport.Capabilities);

	uint32_t ImageCount = SwapChainSupport.Capabilities.minImageCount + 1;
	if (SwapChainSupport.Capabilities.maxImageCount > 0 && ImageCount > SwapChainSupport.Capabilities.maxImageCount)
		ImageCount = SwapChainSupport.Capabilities.maxImageCount;

	VkSwapchainCreateInfoKHR CreateInfo = {};
	CreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	CreateInfo.surface = m_Surface;
	CreateInfo.minImageCount = ImageCount;
	CreateInfo.imageFormat = SurfaceFormat.format;
	CreateInfo.imageColorSpace = SurfaceFormat.colorSpace;
	CreateInfo.imageExtent = Extent;
	CreateInfo.imageArrayLayers = 1;
	CreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices Indices = findQueueFamilies(m_PhysicalDevice);
	uint32_t QueueFamilyIndices[] = { Indices.GraphicsFamily.value(), Indices.PresentFamily.value() };
	if (Indices.GraphicsFamily != Indices.PresentFamily)
	{
		CreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		CreateInfo.queueFamilyIndexCount = 2;
		CreateInfo.pQueueFamilyIndices = QueueFamilyIndices;
	}
	else
	{
		CreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		CreateInfo.queueFamilyIndexCount = 0;
		CreateInfo.pQueueFamilyIndices = nullptr;
	}
	CreateInfo.preTransform = SwapChainSupport.Capabilities.currentTransform;
	CreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	CreateInfo.presentMode = PresentMode;
	CreateInfo.clipped = VK_TRUE;
	CreateInfo.oldSwapchain = VK_NULL_HANDLE;

	if (!IS_VK_SUCCESS(vkCreateSwapchainKHR(m_Device, &CreateInfo, nullptr, &m_SwapChain)))
		VERBOSE_EXIT("failed to create swap chain");

	vkGetSwapchainImagesKHR(m_Device, m_SwapChain, &ImageCount, nullptr);
	m_SwapChainImages.resize(ImageCount);
	vkGetSwapchainImagesKHR(m_Device, m_SwapChain, &ImageCount, m_SwapChainImages.data());
	m_SwapChainImageFormat = SurfaceFormat.format;
	m_SwapChainExtent = Extent;
}

VkSurfaceFormatKHR HelloTriangleApplication::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& vSupportedFormats)
{
	for (const auto& SupportedFormat : vSupportedFormats)
		if (SupportedFormat.format == VK_FORMAT_B8G8R8A8_UNORM && SupportedFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
			return SupportedFormat;
	return vSupportedFormats[0];
}

VkPresentModeKHR HelloTriangleApplication::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& vSupportedPresentModes)
{
	for (const auto& SupportedPresentMode : vSupportedPresentModes)
		if (SupportedPresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			return SupportedPresentMode;
	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D HelloTriangleApplication::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& vCapabilities)
{
	if (vCapabilities.currentExtent.width != UINT32_MAX)
	{
		return vCapabilities.currentExtent;
	}
	else
	{
		VkExtent2D ActualExtent = { WINDOW_WIDTH, WINDOW_HEIGHT };
		ActualExtent.width  = std::max(vCapabilities.minImageExtent.width , std::min(vCapabilities.maxImageExtent.width,  ActualExtent.width ));
		ActualExtent.height = std::max(vCapabilities.minImageExtent.height, std::min(vCapabilities.maxImageExtent.height, ActualExtent.height));
		return ActualExtent;
	}
}

void HelloTriangleApplication::createImageViews()
{
	m_SwapChainImageViews.resize(m_SwapChainImages.size());
	for (size_t i = 0; i < m_SwapChainImages.size(); i++)
	{
		VkImageViewCreateInfo CreateInfo = {};
		CreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		CreateInfo.image = m_SwapChainImages[i];
		CreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		CreateInfo.format = m_SwapChainImageFormat;
		CreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		CreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		CreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		CreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		CreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		CreateInfo.subresourceRange.baseMipLevel = 0;
		CreateInfo.subresourceRange.levelCount = 1;
		CreateInfo.subresourceRange.baseArrayLayer = 0;
		CreateInfo.subresourceRange.layerCount = 1;

		if (!IS_VK_SUCCESS(vkCreateImageView(m_Device, &CreateInfo, nullptr, &m_SwapChainImageViews[i])))
			VERBOSE_EXIT("failed to create image views");
	}
}

void HelloTriangleApplication::createRenderPass()
{
	VkSubpassDependency Dependency = {};
	Dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	Dependency.dstSubpass = 0;
	Dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	Dependency.srcAccessMask = 0;
	Dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	Dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	
	VkAttachmentDescription ColorAttachment = {};
	ColorAttachment.format = m_SwapChainImageFormat;
	ColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	ColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	ColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	ColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	ColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	ColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	ColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference ColorAttachmentRef = {};
	ColorAttachmentRef.attachment = 0;
	ColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription Subpass = {};
	Subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	Subpass.colorAttachmentCount = 1;
	Subpass.pColorAttachments = &ColorAttachmentRef;

	VkRenderPassCreateInfo RenderPassCreateInfo = {};
	RenderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	RenderPassCreateInfo.attachmentCount = 1;
	RenderPassCreateInfo.pAttachments = &ColorAttachment;
	RenderPassCreateInfo.subpassCount = 1;
	RenderPassCreateInfo.pSubpasses = &Subpass;
	RenderPassCreateInfo.dependencyCount = 1;
	RenderPassCreateInfo.pDependencies = &Dependency;

	if (!IS_VK_SUCCESS(vkCreateRenderPass(m_Device, &RenderPassCreateInfo, nullptr, &m_RenderPass)))
		VERBOSE_EXIT("failed to create render pass");
}

void HelloTriangleApplication::createGraphicsPipeline()
{
	// ----- configure user defined shaders ----- //
	auto VertexShaderCode = readFile("Shaders/Triangle.vert.spv");
	auto FragmentShaderCode = readFile("Shaders/Triangle.frag.spv");

	VkShaderModule VertexShaderModule = createShaderModule(VertexShaderCode);
	VkShaderModule FragmentShaderModule = createShaderModule(FragmentShaderCode);

	VkPipelineShaderStageCreateInfo VertexShaderStageInfo = {};
	VertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	VertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	VertexShaderStageInfo.module = VertexShaderModule;
	VertexShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo FragmentShaderStageInfo = {};
	FragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	FragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	FragmentShaderStageInfo.module = FragmentShaderModule;
	FragmentShaderStageInfo.pName = "main";
	
	VkPipelineShaderStageCreateInfo ShaderStages[] = { VertexShaderStageInfo, FragmentShaderStageInfo };
	
	// ----- configure fixed functions ----- //
	VkPipelineVertexInputStateCreateInfo VertexInputInfo = {};
	VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	VertexInputInfo.vertexBindingDescriptionCount = 0;
	VertexInputInfo.pVertexBindingDescriptions = nullptr;
	VertexInputInfo.vertexAttributeDescriptionCount = 0;
	VertexInputInfo.pVertexAttributeDescriptions = nullptr;

	VkPipelineInputAssemblyStateCreateInfo InputAssemblyCreateInfo = {};
	InputAssemblyCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	InputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	InputAssemblyCreateInfo.primitiveRestartEnable = VK_FALSE;

	VkViewport Viewport = {};
	Viewport.x = 0.0f; Viewport.y = 0.0f;
	Viewport.width = (float)m_SwapChainExtent.width; Viewport.height = (float)m_SwapChainExtent.height;
	Viewport.minDepth = 0.0f; Viewport.maxDepth = 1.0f;

	VkRect2D Scissor = {};
	Scissor.offset = { 0, 0 };
	Scissor.extent = m_SwapChainExtent;

	VkPipelineViewportStateCreateInfo ViewportStateCreateInfo = {};
	ViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	ViewportStateCreateInfo.viewportCount = 1;
	ViewportStateCreateInfo.pViewports = &Viewport;
	ViewportStateCreateInfo.scissorCount = 1;
	ViewportStateCreateInfo.pScissors = &Scissor;

	VkPipelineRasterizationStateCreateInfo RasterizerCreateInfo = {};
	RasterizerCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	RasterizerCreateInfo.depthClampEnable = VK_FALSE;
	RasterizerCreateInfo.rasterizerDiscardEnable = VK_FALSE;
	RasterizerCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
	RasterizerCreateInfo.lineWidth = 1.0f;
	RasterizerCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
	RasterizerCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE; // left-hand?
	RasterizerCreateInfo.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo MultisampleCreateInfo = {};
	MultisampleCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	MultisampleCreateInfo.sampleShadingEnable = VK_FALSE;
	MultisampleCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState ColorBlendAttachmentState = {};
	ColorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	ColorBlendAttachmentState.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo ColorBlendingCreateInfo = {};
	ColorBlendingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	ColorBlendingCreateInfo.logicOpEnable = VK_FALSE;
	ColorBlendingCreateInfo.logicOp = VK_LOGIC_OP_COPY;
	ColorBlendingCreateInfo.attachmentCount = 1;
	ColorBlendingCreateInfo.attachmentCount = 1;
	ColorBlendingCreateInfo.pAttachments = &ColorBlendAttachmentState;
	ColorBlendingCreateInfo.blendConstants[0] = 0.0f;
	ColorBlendingCreateInfo.blendConstants[1] = 0.0f;
	ColorBlendingCreateInfo.blendConstants[2] = 0.0f;
	ColorBlendingCreateInfo.blendConstants[3] = 0.0f;

	VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo = {};
	PipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	PipelineLayoutCreateInfo.setLayoutCount = 0;
	PipelineLayoutCreateInfo.pushConstantRangeCount = 0;

	if (!IS_VK_SUCCESS(vkCreatePipelineLayout(m_Device, &PipelineLayoutCreateInfo, nullptr, &m_PipelineLayout)))
		VERBOSE_EXIT("failed to create pipeline layout");

	VkGraphicsPipelineCreateInfo PipelineCreateInfo = {};
	PipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	PipelineCreateInfo.stageCount = 2;
	PipelineCreateInfo.pStages = ShaderStages;
	PipelineCreateInfo.pVertexInputState = &VertexInputInfo;
	PipelineCreateInfo.pInputAssemblyState = &InputAssemblyCreateInfo;
	PipelineCreateInfo.pViewportState = &ViewportStateCreateInfo;
	PipelineCreateInfo.pRasterizationState = &RasterizerCreateInfo;
	PipelineCreateInfo.pMultisampleState = &MultisampleCreateInfo;
	PipelineCreateInfo.pDepthStencilState = nullptr;
	PipelineCreateInfo.pColorBlendState = &ColorBlendingCreateInfo;
	PipelineCreateInfo.pDynamicState = nullptr;
	PipelineCreateInfo.layout = m_PipelineLayout;
	PipelineCreateInfo.renderPass = m_RenderPass;
	PipelineCreateInfo.subpass = 0;
	PipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	PipelineCreateInfo.basePipelineIndex = -1;

	if (!IS_VK_SUCCESS(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &PipelineCreateInfo, nullptr, &m_GraphicsPipeline)))
		VERBOSE_EXIT("failed to create graphics pipeline");

	vkDestroyShaderModule(m_Device, VertexShaderModule, nullptr);
	vkDestroyShaderModule(m_Device, FragmentShaderModule, nullptr);
}

VkShaderModule HelloTriangleApplication::createShaderModule(const std::vector<char>& vCode)
{
	VkShaderModuleCreateInfo CreateInfo = {};
	CreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	CreateInfo.codeSize = vCode.size();
	CreateInfo.pCode = reinterpret_cast<const uint32_t*>(vCode.data());

	VkShaderModule ShaderModule;
	if (!IS_VK_SUCCESS(vkCreateShaderModule(m_Device, &CreateInfo, nullptr, &ShaderModule)))
		VERBOSE_EXIT("failed to create shader module");
	return ShaderModule;
}

void HelloTriangleApplication::createFrameBuffers()
{
	m_SwapChainFrameBuffers.resize(m_SwapChainImageViews.size());
	for (size_t i = 0; i < m_SwapChainImageViews.size(); i++)
	{
		VkImageView Attachments[] = { m_SwapChainImageViews[i] };
		VkFramebufferCreateInfo FrameBufferCreateInfo = {};
		FrameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		FrameBufferCreateInfo.renderPass = m_RenderPass;
		FrameBufferCreateInfo.attachmentCount = 1;
		FrameBufferCreateInfo.pAttachments = Attachments;
		FrameBufferCreateInfo.width = m_SwapChainExtent.width;
		FrameBufferCreateInfo.height = m_SwapChainExtent.height;
		FrameBufferCreateInfo.layers = 1;

		if (!IS_VK_SUCCESS(vkCreateFramebuffer(m_Device, &FrameBufferCreateInfo, nullptr, &m_SwapChainFrameBuffers[i])))
			VERBOSE_EXIT("failed to create frame buffer");
	}
}

void HelloTriangleApplication::createCommandPool()
{
	QueueFamilyIndices QueueFamilyIndices = findQueueFamilies(m_PhysicalDevice);

	VkCommandPoolCreateInfo CommandPoolCreateInfo = {};
	CommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	CommandPoolCreateInfo.queueFamilyIndex = QueueFamilyIndices.GraphicsFamily.value();

	if (!IS_VK_SUCCESS(vkCreateCommandPool(m_Device, &CommandPoolCreateInfo, nullptr, &m_CommandPool)))
		VERBOSE_EXIT("failed to create command pool");
}

void HelloTriangleApplication::createCommandBuffers()
{
	m_CommandBuffers.resize(m_SwapChainFrameBuffers.size());
	
	VkCommandBufferAllocateInfo AllocInfo = {};
	AllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	AllocInfo.commandPool = m_CommandPool;
	AllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	AllocInfo.commandBufferCount = (uint32_t)m_CommandBuffers.size();

	if (!IS_VK_SUCCESS(vkAllocateCommandBuffers(m_Device, &AllocInfo, m_CommandBuffers.data())))
		VERBOSE_EXIT("failed to allocate command buffers");

	for (size_t i = 0; i < m_CommandBuffers.size(); i++)
	{
		VkCommandBufferBeginInfo BeginInfo = {};
		BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (!IS_VK_SUCCESS(vkBeginCommandBuffer(m_CommandBuffers[i], &BeginInfo)))
			VERBOSE_EXIT("failed to begin recording command buffer");

		VkRenderPassBeginInfo RenderPassBeginInfo = {};
		RenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		RenderPassBeginInfo.renderPass = m_RenderPass;
		RenderPassBeginInfo.framebuffer = m_SwapChainFrameBuffers[i];
		RenderPassBeginInfo.renderArea.offset = { 0, 0 };
		RenderPassBeginInfo.renderArea.extent = m_SwapChainExtent;

		VkClearValue ClearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		RenderPassBeginInfo.clearValueCount = 1;
		RenderPassBeginInfo.pClearValues = &ClearColor;

		vkCmdBeginRenderPass(m_CommandBuffers[i], &RenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(m_CommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_GraphicsPipeline);
		vkCmdDraw(m_CommandBuffers[i], 3, 1, 0, 0);
		vkCmdEndRenderPass(m_CommandBuffers[i]);

		if (!IS_VK_SUCCESS(vkEndCommandBuffer(m_CommandBuffers[i])))
			VERBOSE_EXIT("failed to record command buffer");
	}
}

void HelloTriangleApplication::createSyncObjects()
{
	m_ImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	m_RenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	m_InFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
	
	VkSemaphoreCreateInfo SemaphoreCreateInfo = {};
	SemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo FenceCreateInfo = {};
	FenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	FenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (!IS_VK_SUCCESS(vkCreateSemaphore(m_Device, &SemaphoreCreateInfo, nullptr, &m_ImageAvailableSemaphores[i]))
			|| !IS_VK_SUCCESS(vkCreateSemaphore(m_Device, &SemaphoreCreateInfo, nullptr, &m_RenderFinishedSemaphores[i]))
			|| vkCreateFence(m_Device, &FenceCreateInfo, nullptr, &m_InFlightFences[i]))
			VERBOSE_EXIT("failed to create semaphores");
	}
}

void HelloTriangleApplication::mainLoop()
{
	while (!glfwWindowShouldClose(m_pWindow))
	{
		glfwPollEvents();
		drawFrame();
	}

	vkDeviceWaitIdle(m_Device);
}

void HelloTriangleApplication::drawFrame()
{
	vkWaitForFences(m_Device, 1, &m_InFlightFences[m_CurrentFrame], VK_TRUE, UINT64_MAX);
	vkResetFences(m_Device, 1, &m_InFlightFences[m_CurrentFrame]);
	
	uint32_t ImageIndex;
	vkAcquireNextImageKHR(m_Device, m_SwapChain, UINT64_MAX, m_ImageAvailableSemaphores[m_CurrentFrame], VK_NULL_HANDLE, &ImageIndex);

	VkSubmitInfo SubmitInfo = {};
	SubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore WaitSemaphores[] = { m_ImageAvailableSemaphores[m_CurrentFrame] };
	VkPipelineStageFlags WaitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	SubmitInfo.waitSemaphoreCount = 1;
	SubmitInfo.pWaitSemaphores = WaitSemaphores;
	SubmitInfo.pWaitDstStageMask = WaitStages;
	SubmitInfo.commandBufferCount = 1;
	SubmitInfo.pCommandBuffers = &m_CommandBuffers[ImageIndex];

	VkSemaphore SignalSemaphores[] = { m_RenderFinishedSemaphores[m_CurrentFrame] };
	SubmitInfo.signalSemaphoreCount = 1;
	SubmitInfo.pSignalSemaphores = SignalSemaphores;

	if (!IS_VK_SUCCESS(vkQueueSubmit(m_GraphicsQueue, 1, &SubmitInfo, m_InFlightFences[m_CurrentFrame])))
		VERBOSE_EXIT("failed to submit draw command buffer");

	VkPresentInfoKHR PresentInfo = {};
	PresentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	PresentInfo.waitSemaphoreCount = 1;
	PresentInfo.pWaitSemaphores = SignalSemaphores;

	VkSwapchainKHR SwapChains[] = { m_SwapChain };
	PresentInfo.swapchainCount = 1;
	PresentInfo.pSwapchains = SwapChains;
	PresentInfo.pImageIndices = &ImageIndex;
	vkQueuePresentKHR(m_PresentQueue, &PresentInfo);

	m_CurrentFrame = (m_CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void HelloTriangleApplication::cleanup()
{
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
	{
		vkDestroySemaphore(m_Device, m_RenderFinishedSemaphores[i], nullptr);
		vkDestroySemaphore(m_Device, m_ImageAvailableSemaphores[i], nullptr);
		vkDestroyFence(m_Device, m_InFlightFences[i], nullptr);
	}
	vkDestroyCommandPool(m_Device, m_CommandPool, nullptr);
	for (auto FrameBuffer : m_SwapChainFrameBuffers)
		vkDestroyFramebuffer(m_Device, FrameBuffer, nullptr);
	
	vkDestroyPipeline(m_Device, m_GraphicsPipeline, nullptr);
	vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
	vkDestroyRenderPass(m_Device, m_RenderPass, nullptr);
	for (auto ImageView : m_SwapChainImageViews)
		vkDestroyImageView(m_Device, ImageView, nullptr);
	vkDestroySwapchainKHR(m_Device, m_SwapChain, nullptr);
	vkDestroyDevice(m_Device, nullptr);
	if constexpr (EnableValidationLayers) DestroyDebugUtilsMessengerEXT(m_Instance, m_DebugMessenger, nullptr);
	vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
	vkDestroyInstance(m_Instance, nullptr);
	
	glfwDestroyWindow(m_pWindow);
	glfwTerminate();
}
