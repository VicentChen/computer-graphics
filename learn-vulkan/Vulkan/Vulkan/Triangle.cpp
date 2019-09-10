#include "Triangle.h"
#include <algorithm>
#include <set>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <unordered_map>

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

void HelloTriangleApplication::framebufferResizeCallback(GLFWwindow* vWindow, int vWidth, int vHeight)
{
	auto pApp = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(vWindow));
	pApp->m_FramebufferResized = true;
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
	m_pWindow = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan", nullptr, nullptr);
	glfwSetWindowUserPointer(m_pWindow, this);
	glfwSetFramebufferSizeCallback(m_pWindow, framebufferResizeCallback);
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
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createCommandPool();
	createColorResources();
	createDepthResources();
	createFrameBuffers();
	createTextureImage();
	createTextureImageView();
	createTextureSampler();
	loadModel();
	createVertexBuffer();
	createIndexBuffer();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
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
			m_MSAASamples = getMaxUsableSampleCount();
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

	VkPhysicalDeviceFeatures SupportedFeatures;
	vkGetPhysicalDeviceFeatures(vDevice, &SupportedFeatures);
	
	return Indices.isComplete() && ExtensionSupported && SwapChainAdequate && SupportedFeatures.samplerAnisotropy;
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

VkSampleCountFlagBits HelloTriangleApplication::getMaxUsableSampleCount()
{
	VkPhysicalDeviceProperties PhysicalDeviceProperties;
	vkGetPhysicalDeviceProperties(m_PhysicalDevice, &PhysicalDeviceProperties);

	VkSampleCountFlags Counts = std::min(PhysicalDeviceProperties.limits.framebufferColorSampleCounts, PhysicalDeviceProperties.limits.framebufferDepthSampleCounts);
	if (Counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
	if (Counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
	if (Counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
	if (Counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
	if (Counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
	if (Counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }
	return VK_SAMPLE_COUNT_1_BIT;
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
	DeviceFeatures.samplerAnisotropy = VK_TRUE;
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
		int Width, Height;
		glfwGetFramebufferSize(m_pWindow, &Width, &Height);
		
		VkExtent2D ActualExtent = { static_cast<uint32_t>(Width), static_cast<uint32_t>(Height) };
		ActualExtent.width  = std::max(vCapabilities.minImageExtent.width , std::min(vCapabilities.maxImageExtent.width,  ActualExtent.width ));
		ActualExtent.height = std::max(vCapabilities.minImageExtent.height, std::min(vCapabilities.maxImageExtent.height, ActualExtent.height));
		return ActualExtent;
	}
}

void HelloTriangleApplication::createImageViews()
{
	m_SwapChainImageViews.resize(m_SwapChainImages.size());
	for (size_t i = 0; i < m_SwapChainImages.size(); i++)
		m_SwapChainImageViews[i] = createImageView(m_SwapChainImages[i], m_SwapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
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
	ColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	ColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	ColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	ColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	ColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	ColorAttachment.samples = m_MSAASamples;
	ColorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference ColorAttachmentRef = {};
	ColorAttachmentRef.attachment = 0;
	ColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription DepthAttachment = {};
	DepthAttachment.format = findDepthFormat();
	DepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	DepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	DepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	DepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	DepthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	DepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	DepthAttachment.samples = m_MSAASamples;

	VkAttachmentReference DepthAttachmentRef = {};
	DepthAttachmentRef.attachment = 1;
	DepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription ColorAttachmentResolve = {};
	ColorAttachmentResolve.format = m_SwapChainImageFormat;
	ColorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
	ColorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	ColorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	ColorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	ColorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	ColorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	ColorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	
	VkAttachmentReference ColorAttachmentResolveRef = {};
	ColorAttachmentResolveRef.attachment = 2;
	ColorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	
	VkSubpassDescription Subpass = {};
	Subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	Subpass.colorAttachmentCount = 1;
	Subpass.pColorAttachments = &ColorAttachmentRef;
	Subpass.pDepthStencilAttachment = &DepthAttachmentRef;
	Subpass.pResolveAttachments = &ColorAttachmentResolveRef;

	std::array<VkAttachmentDescription, 3> Attachments = { ColorAttachment, DepthAttachment, ColorAttachmentResolve };
	
	VkRenderPassCreateInfo RenderPassCreateInfo = {};
	RenderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	RenderPassCreateInfo.attachmentCount = static_cast<uint32_t>(Attachments.size());
	RenderPassCreateInfo.pAttachments = Attachments.data();
	RenderPassCreateInfo.subpassCount = 1;
	RenderPassCreateInfo.pSubpasses = &Subpass;
	RenderPassCreateInfo.dependencyCount = 1;
	RenderPassCreateInfo.pDependencies = &Dependency;

	if (!IS_VK_SUCCESS(vkCreateRenderPass(m_Device, &RenderPassCreateInfo, nullptr, &m_RenderPass)))
		VERBOSE_EXIT("failed to create render pass");
}

void HelloTriangleApplication::createDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding UboLayoutBinding = {};
	UboLayoutBinding.binding = 0;
	UboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	UboLayoutBinding.descriptorCount = 1;
	UboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	UboLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding SamplerLayoutBinding = {};
	SamplerLayoutBinding.binding = 1;
	SamplerLayoutBinding.descriptorCount = 1;
	SamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	SamplerLayoutBinding.pImmutableSamplers = nullptr;
	SamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	std::array<VkDescriptorSetLayoutBinding, 2> Bindings = { UboLayoutBinding, SamplerLayoutBinding };
	
	VkDescriptorSetLayoutCreateInfo LayoutCreateInfo = {};
	LayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	LayoutCreateInfo.bindingCount = static_cast<uint32_t>(Bindings.size());
	LayoutCreateInfo.pBindings = Bindings.data();
	if (!IS_VK_SUCCESS(vkCreateDescriptorSetLayout(m_Device, &LayoutCreateInfo, nullptr, &m_DescriptorSetLayout)))
		VERBOSE_EXIT("failed to create descriptor set layout");
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
	auto BindingDescription = Vertex::getBindingDescription();
	auto AttributeDescription = Vertex::getAttributeDescriptions();
	VkPipelineVertexInputStateCreateInfo VertexInputInfo = {};
	VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	VertexInputInfo.vertexBindingDescriptionCount = 1;
	VertexInputInfo.pVertexBindingDescriptions = &BindingDescription;
	VertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(AttributeDescription.size());
	VertexInputInfo.pVertexAttributeDescriptions = AttributeDescription.data();

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
	RasterizerCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // left-hand?
	RasterizerCreateInfo.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo MultisampleCreateInfo = {};
	MultisampleCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	MultisampleCreateInfo.sampleShadingEnable = VK_FALSE;
	MultisampleCreateInfo.rasterizationSamples = m_MSAASamples;

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
	PipelineLayoutCreateInfo.setLayoutCount = 1;
	PipelineLayoutCreateInfo.pSetLayouts = &m_DescriptorSetLayout;
	PipelineLayoutCreateInfo.pushConstantRangeCount = 0;

	if (!IS_VK_SUCCESS(vkCreatePipelineLayout(m_Device, &PipelineLayoutCreateInfo, nullptr, &m_PipelineLayout)))
		VERBOSE_EXIT("failed to create pipeline layout");

	VkPipelineDepthStencilStateCreateInfo DepthStencilCreateInfo = {};
	DepthStencilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	DepthStencilCreateInfo.depthTestEnable = VK_TRUE;
	DepthStencilCreateInfo.depthWriteEnable = VK_TRUE;
	DepthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
	DepthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE;
	DepthStencilCreateInfo.minDepthBounds = 0.0f;
	DepthStencilCreateInfo.maxDepthBounds = 1.0f;
	DepthStencilCreateInfo.stencilTestEnable = VK_FALSE;
	DepthStencilCreateInfo.front = {};
	DepthStencilCreateInfo.back = {};
	
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
	PipelineCreateInfo.pDepthStencilState = &DepthStencilCreateInfo;

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
		std::array<VkImageView, 3> Attachments = { m_ColorImageView, m_DepthImageView, m_SwapChainImageViews[i] };
		VkFramebufferCreateInfo FrameBufferCreateInfo = {};
		FrameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		FrameBufferCreateInfo.renderPass = m_RenderPass;
		FrameBufferCreateInfo.attachmentCount = static_cast<uint32_t>(Attachments.size());
		FrameBufferCreateInfo.pAttachments = Attachments.data();
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

void HelloTriangleApplication::createColorResources()
{
	VkFormat ColorFormat = m_SwapChainImageFormat;

	createImage(m_SwapChainExtent.width, m_SwapChainExtent.height, 1, m_MSAASamples, ColorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_ColorImage, m_ColorImageMemory);
	m_ColorImageView = createImageView(m_ColorImage, ColorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	transitionImageLayout(m_ColorImage, ColorFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);
}

void HelloTriangleApplication::createDepthResources()
{
	VkFormat DepthFormat = findDepthFormat();
	createImage(m_SwapChainExtent.width, m_SwapChainExtent.height, 1, m_MSAASamples, DepthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_DepthImage, m_DepthImageMemory);
	m_DepthImageView = createImageView(m_DepthImage, DepthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
	transitionImageLayout(m_DepthImage, DepthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
}

bool HelloTriangleApplication::hasStencilComponent(enum VkFormat vFormat)
{
	return vFormat == VK_FORMAT_D32_SFLOAT_S8_UINT || vFormat == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat HelloTriangleApplication::findDepthFormat()
{
	return findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkFormat HelloTriangleApplication::findSupportedFormat(const std::vector<VkFormat>& vCandidates, VkImageTiling vTiling, VkFormatFeatureFlags vFeatures)
{
	for (VkFormat Format : vCandidates)
	{
		VkFormatProperties Props;
		vkGetPhysicalDeviceFormatProperties(m_PhysicalDevice, Format, &Props);
		if (vTiling == VK_IMAGE_TILING_LINEAR && (Props.linearTilingFeatures & vFeatures) == vFeatures) return Format;
		else if (vTiling == VK_IMAGE_TILING_OPTIMAL && (Props.optimalTilingFeatures & vFeatures) == vFeatures) return Format;		
	}
	VERBOSE_EXIT("failed to find supported format");
}

void HelloTriangleApplication::createTextureImage()
{
	int TexWidth, TexHeight, TexChannels;
	stbi_uc* Pixels = stbi_load(TEXTURE_PATH.c_str(), &TexWidth, &TexHeight, &TexChannels, STBI_rgb_alpha);
	VkDeviceSize ImageSize = TexWidth * TexHeight * 4;
	if (!Pixels) VERBOSE_EXIT("failed to load texture image");
	m_MipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(TexWidth, TexHeight)))) + 1;
	
	VkBuffer StagingBuffer;
	VkDeviceMemory StagingBufferMemory;
	createBuffer(ImageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, StagingBuffer, StagingBufferMemory);
	void *Data;
	vkMapMemory(m_Device, StagingBufferMemory, 0, ImageSize, 0, &Data);
	memcpy(Data, Pixels, static_cast<size_t>(ImageSize));
	vkUnmapMemory(m_Device, StagingBufferMemory);
	stbi_image_free(Pixels);

	createImage(TexWidth, TexHeight, m_MipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_TextureImage, m_TextureImageMemory);
	transitionImageLayout(m_TextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, m_MipLevels);
	copyBufferToImage(StagingBuffer, m_TextureImage, static_cast<uint32_t>(TexWidth), static_cast<uint32_t>(TexHeight));

	vkDestroyBuffer(m_Device, StagingBuffer, nullptr);
	vkFreeMemory(m_Device, StagingBufferMemory, nullptr);
	
	generateMipmaps(m_TextureImage, VK_FORMAT_R8G8B8A8_UNORM, TexWidth, TexHeight, m_MipLevels);
}

void HelloTriangleApplication::generateMipmaps(VkImage vImage, VkFormat vFormat, int32_t vTexWidth, int32_t vTexHeight, uint32_t vMipLevels)
{
	VkFormatProperties FormatProperties;
	vkGetPhysicalDeviceFormatProperties(m_PhysicalDevice, vFormat, &FormatProperties);
	if (!(FormatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
		VERBOSE_EXIT("texture image format does not support linear blitting");
	
	VkCommandBuffer CommandBuffer = beginSingleTimeCommands();
	VkImageMemoryBarrier Barrier = {};
	Barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	Barrier.image = vImage;
	Barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	Barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	Barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	Barrier.subresourceRange.baseArrayLayer = 0;
	Barrier.subresourceRange.layerCount = 1;
	Barrier.subresourceRange.levelCount = 1;

	int32_t MipWidth = vTexWidth;
	int32_t MipHeight = vTexHeight;

	for (uint32_t i = 1; i < vMipLevels; i++)
	{
		Barrier.subresourceRange.baseMipLevel = i - 1;
		Barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		Barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		Barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		Barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &Barrier);

		VkImageBlit Blit = {};
		Blit.srcOffsets[0] = { 0, 0, 0 };
		Blit.srcOffsets[1] = { MipWidth, MipHeight, 1 };
		Blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		Blit.srcSubresource.mipLevel = i - 1;
		Blit.srcSubresource.baseArrayLayer = 0;
		Blit.srcSubresource.layerCount = 1;
		Blit.dstOffsets[0] = { 0, 0, 0 };
		Blit.dstOffsets[1] = { MipWidth > 1 ? MipWidth / 2 : 1, MipHeight > 1 ? MipHeight / 2 : 1, 1 };
		Blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		Blit.dstSubresource.mipLevel = i;
		Blit.dstSubresource.baseArrayLayer = 0;
		Blit.dstSubresource.layerCount = 1;

		vkCmdBlitImage(CommandBuffer, vImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &Blit, VK_FILTER_LINEAR);

		Barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		Barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		Barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		Barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &Barrier);

		if (MipWidth > 1) MipWidth /= 2;
		if (MipHeight > 1) MipHeight /= 2;
	}

	Barrier.subresourceRange.baseMipLevel = m_MipLevels - 1;
	Barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	Barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	Barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	Barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &Barrier);
	
	endSingleTimeCommands(CommandBuffer);
}

void HelloTriangleApplication::createTextureImageView()
{
	m_TextureImageView = createImageView(m_TextureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, m_MipLevels);
}

void HelloTriangleApplication::createTextureSampler()
{
	VkSamplerCreateInfo SamplerInfo = {};
	SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	SamplerInfo.magFilter = VK_FILTER_LINEAR;
	SamplerInfo.minFilter = VK_FILTER_LINEAR;
	SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	SamplerInfo.anisotropyEnable = VK_TRUE;
	SamplerInfo.maxAnisotropy = 16;
	SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	SamplerInfo.unnormalizedCoordinates = VK_FALSE;
	SamplerInfo.compareEnable = VK_FALSE;
	SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	SamplerInfo.mipLodBias = 0.0f;
	SamplerInfo.minLod = 0.0f;
	SamplerInfo.maxLod = static_cast<float>(m_MipLevels);

	if (!IS_VK_SUCCESS(vkCreateSampler(m_Device, &SamplerInfo, nullptr, &m_TextureSampler)))
		VERBOSE_EXIT("failed to create texture sampler");
}

VkImageView HelloTriangleApplication::createImageView(VkImage vImage, VkFormat vFormat, VkImageAspectFlags vAspectFlags, uint32_t vMipLevels)
{
	VkImageViewCreateInfo ViewInfo = {};
	ViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	ViewInfo.image = vImage;
	ViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	ViewInfo.format = vFormat;
	ViewInfo.subresourceRange.aspectMask = vAspectFlags;
	ViewInfo.subresourceRange.baseMipLevel = 0;
	ViewInfo.subresourceRange.levelCount = vMipLevels;
	ViewInfo.subresourceRange.baseArrayLayer = 0;
	ViewInfo.subresourceRange.layerCount = 1;

	VkImageView ImageView;
	if (!IS_VK_SUCCESS(vkCreateImageView(m_Device, &ViewInfo, nullptr, &ImageView)))
		VERBOSE_EXIT("failed to create textur image view");
	return ImageView;
}

void HelloTriangleApplication::copyBufferToImage(VkBuffer vBuffer, VkImage vImage, uint32_t vWidth, uint32_t vHeight)
{
	VkCommandBuffer CommandBuffer = beginSingleTimeCommands();
	VkBufferImageCopy Region = {};
	Region.bufferOffset = 0;
	Region.bufferRowLength = 0;
	Region.bufferImageHeight = 0;
	Region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	Region.imageSubresource.mipLevel = 0;
	Region.imageSubresource.baseArrayLayer = 0;
	Region.imageSubresource.layerCount = 1;
	Region.imageOffset = { 0, 0, 0 };
	Region.imageExtent = { vWidth, vHeight, 1 };
	vkCmdCopyBufferToImage(CommandBuffer, vBuffer, vImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &Region);
	endSingleTimeCommands(CommandBuffer);
}

VkCommandBuffer HelloTriangleApplication::beginSingleTimeCommands()
{
	VkCommandBufferAllocateInfo AllocInfo = {};
	AllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	AllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	AllocInfo.commandPool = m_CommandPool;
	AllocInfo.commandBufferCount = 1;

	VkCommandBuffer CommandBuffer;
	vkAllocateCommandBuffers(m_Device, &AllocInfo, &CommandBuffer);
	VkCommandBufferBeginInfo BeginInfo = {};
	BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	BeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(CommandBuffer, &BeginInfo);
	
	return CommandBuffer;
}

void HelloTriangleApplication::endSingleTimeCommands(VkCommandBuffer vCommandBuffer)
{
	vkEndCommandBuffer(vCommandBuffer);
	VkSubmitInfo SubmitInfo = {};
	SubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	SubmitInfo.commandBufferCount = 1;
	SubmitInfo.pCommandBuffers = &vCommandBuffer;
	vkQueueSubmit(m_GraphicsQueue, 1, &SubmitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(m_GraphicsQueue);

	vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &vCommandBuffer);
}

void HelloTriangleApplication::transitionImageLayout(VkImage vImage, VkFormat vFormat, VkImageLayout vOldLayout, VkImageLayout vNewLayout, uint32_t vMipLevels)
{
	VkCommandBuffer CommandBuffer = beginSingleTimeCommands();
	VkImageMemoryBarrier Barrier = {};
	Barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	Barrier.oldLayout = vOldLayout;
	Barrier.newLayout = vNewLayout;
	Barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	Barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	Barrier.image = vImage;
	Barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	Barrier.subresourceRange.baseMipLevel = 0;
	Barrier.subresourceRange.levelCount = vMipLevels;
	Barrier.subresourceRange.baseArrayLayer = 0;
	Barrier.subresourceRange.layerCount = 1;
	Barrier.srcAccessMask = 0;
	Barrier.dstAccessMask = 0;

	VkPipelineStageFlags SourceStage;
	VkPipelineStageFlags DestinationStage;

	if (vOldLayout == VK_IMAGE_LAYOUT_UNDEFINED && vNewLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		Barrier.srcAccessMask = 0;
		Barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		SourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		DestinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (vOldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && vNewLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		Barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		Barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		SourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		DestinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else if (vOldLayout == VK_IMAGE_LAYOUT_UNDEFINED && vNewLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
	{
		Barrier.srcAccessMask = 0;
		Barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		SourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		DestinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	}
	else if (vOldLayout == VK_IMAGE_LAYOUT_UNDEFINED && vNewLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
	{
		Barrier.srcAccessMask = 0;
		Barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	}
	else
	{
		VERBOSE_EXIT("unsupported layout transition");
	}

	if (vNewLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
	{
		Barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		if (hasStencilComponent(vFormat)) Barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
	}
	else
	{
		Barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}
	
	vkCmdPipelineBarrier(
		CommandBuffer,
		SourceStage, DestinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &Barrier
	);
	
	endSingleTimeCommands(CommandBuffer);
}

void HelloTriangleApplication::createImage(uint32_t vWidth, uint32_t vHeight, uint32_t vMipLevels, VkSampleCountFlagBits vNumSamples, VkFormat vFormat, VkImageTiling vTiling, VkImageUsageFlags vUsage, VkMemoryPropertyFlags vProperties, VkImage& vImage, VkDeviceMemory& vImageMemory)
{
	VkImageCreateInfo ImageInfo = {};
	ImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ImageInfo.imageType = VK_IMAGE_TYPE_2D;
	ImageInfo.extent.width = vWidth;
	ImageInfo.extent.height = vHeight;
	ImageInfo.extent.depth = 1;
	ImageInfo.mipLevels = vMipLevels;
	ImageInfo.arrayLayers = 1;
	ImageInfo.format = vFormat;
	ImageInfo.tiling = vTiling;
	ImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	ImageInfo.usage = vUsage;
	ImageInfo.samples = vNumSamples;
	ImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	if (!IS_VK_SUCCESS(vkCreateImage(m_Device, &ImageInfo, nullptr, &vImage)))
		VERBOSE_EXIT("failed to create image");

	VkMemoryRequirements MemRequirements;
	vkGetImageMemoryRequirements(m_Device, vImage, &MemRequirements);
	VkMemoryAllocateInfo AllocInfo = {};
	AllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	AllocInfo.allocationSize = MemRequirements.size;
	AllocInfo.memoryTypeIndex = findMemoryType(MemRequirements.memoryTypeBits, vProperties);

	if (!IS_VK_SUCCESS(vkAllocateMemory(m_Device, &AllocInfo, nullptr, &vImageMemory)))
		VERBOSE_EXIT("failed to allocate vertex buffer memory");

	vkBindImageMemory(m_Device, vImage, vImageMemory, 0);
}

void HelloTriangleApplication::loadModel()
{
	tinyobj::attrib_t Attrib;
	std::vector<tinyobj::shape_t> Shapes;
	std::vector<tinyobj::material_t> Materials;
	std::string Warn, Err;
	if (!tinyobj::LoadObj(&Attrib, &Shapes, &Materials, &Warn, &Err, MODEL_PATH.c_str()))
		VERBOSE_EXIT(Warn + Err);

	std::unordered_map<Vertex, uint32_t> UniqueVertices = {};
	
	for (const auto& Shape : Shapes)
	{
		for (const auto& Index : Shape.mesh.indices)
		{
			Vertex Vert = {};
			Vert.Pos = {
				Attrib.vertices[3 * Index.vertex_index + 0],
				Attrib.vertices[3 * Index.vertex_index + 1],
				Attrib.vertices[3 * Index.vertex_index + 2],
			};
			Vert.TexCoord = {
				Attrib.texcoords[2 * Index.texcoord_index + 0],
				1.0f - Attrib.texcoords[2 * Index.texcoord_index + 1],
			};
			Vert.Color = { 1.0f, 1.0f, 1.0f };

			if (UniqueVertices.count(Vert) == 0) {
				UniqueVertices[Vert] = static_cast<uint32_t>(m_Vertices.size());
				m_Vertices.push_back(Vert);
			}
			m_Indices.push_back(UniqueVertices[Vert]);
		}
	}
}

void HelloTriangleApplication::createVertexBuffer()
{
	VkDeviceSize BufferSize = sizeof(m_Vertices[0]) * m_Vertices.size();
	void *Data;

	VkBuffer StagingBuffer;
	VkDeviceMemory StagingBufferMemory;
	createBuffer(BufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, StagingBuffer, StagingBufferMemory);	
	vkMapMemory(m_Device, StagingBufferMemory, 0, BufferSize, 0, &Data);
	memcpy(Data, m_Vertices.data(), BufferSize);
	vkUnmapMemory(m_Device, StagingBufferMemory);
	
	createBuffer(BufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_VertexBuffer, m_VertexBufferMemory);
	copyBuffer(StagingBuffer, m_VertexBuffer, BufferSize);

	vkDestroyBuffer(m_Device, StagingBuffer, nullptr);
	vkFreeMemory(m_Device, StagingBufferMemory, nullptr);
}

uint32_t HelloTriangleApplication::findMemoryType(uint32_t vTypeFilter, VkMemoryPropertyFlags vProperties)
{
	VkPhysicalDeviceMemoryProperties MemProperties;
	vkGetPhysicalDeviceMemoryProperties(m_PhysicalDevice, &MemProperties);
	for (uint32_t i = 0; i < MemProperties.memoryTypeCount; i++) {
		if ((vTypeFilter & (1 << i)) && (MemProperties.memoryTypes[i].propertyFlags & vProperties) == vProperties) return i;
	}
	VERBOSE_EXIT("failed to find suitable memory type");
}

void HelloTriangleApplication::createBuffer(VkDeviceSize vSize, VkBufferUsageFlags vUsage, VkMemoryPropertyFlags vProperties, VkBuffer& vBuffer, VkDeviceMemory& vBufferMemory)
{
	VkBufferCreateInfo BufferInfo = {};
	BufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	BufferInfo.size = vSize;
	BufferInfo.usage = vUsage;
	BufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (!IS_VK_SUCCESS(vkCreateBuffer(m_Device, &BufferInfo, nullptr, &vBuffer)))
		VERBOSE_EXIT("failed to create vertex buffer");

	VkMemoryRequirements MemRequirements;
	vkGetBufferMemoryRequirements(m_Device, vBuffer, &MemRequirements);
	VkMemoryAllocateInfo AllocInfo = {};
	AllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	AllocInfo.allocationSize = MemRequirements.size;
	AllocInfo.memoryTypeIndex = findMemoryType(MemRequirements.memoryTypeBits, vProperties);

	if (!IS_VK_SUCCESS(vkAllocateMemory(m_Device, &AllocInfo, nullptr, &vBufferMemory)))
		VERBOSE_EXIT("failed to allocate vertex buffer memory");

	vkBindBufferMemory(m_Device, vBuffer, vBufferMemory, 0);
}

void HelloTriangleApplication::copyBuffer(VkBuffer vSrcBuffer, VkBuffer vDstBuffer, VkDeviceSize vSize)
{
	VkCommandBuffer CommandBuffer = beginSingleTimeCommands();
	VkBufferCopy CopyRegion = {};
	CopyRegion.size = vSize;
	vkCmdCopyBuffer(CommandBuffer, vSrcBuffer, vDstBuffer, 1, &CopyRegion);
	endSingleTimeCommands(CommandBuffer);
}

void HelloTriangleApplication::createIndexBuffer()
{
	VkDeviceSize BufferSize = sizeof(m_Indices[0]) * m_Indices.size();
	
	void *Data;

	VkBuffer StagingBuffer;
	VkDeviceMemory StagingBufferMemory;
	createBuffer(BufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, StagingBuffer, StagingBufferMemory);
	vkMapMemory(m_Device, StagingBufferMemory, 0, BufferSize, 0, &Data);
	memcpy(Data, m_Indices.data(), BufferSize);
	vkUnmapMemory(m_Device, StagingBufferMemory);

	createBuffer(BufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_IndexBuffer, m_IndexBufferMemory);
	copyBuffer(StagingBuffer, m_IndexBuffer, BufferSize);
	
	vkDestroyBuffer(m_Device, StagingBuffer, nullptr);
	vkFreeMemory(m_Device, StagingBufferMemory, nullptr);
}

void HelloTriangleApplication::createUniformBuffers()
{
	VkDeviceSize BufferSize = sizeof(UniformBufferObject);
	m_UniformBuffers.resize(m_SwapChainImages.size());
	m_UniformBufferMemory.resize(m_SwapChainImages.size());

	for (size_t i = 0; i < m_SwapChainImages.size(); i++)
		createBuffer(BufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_UniformBuffers[i], m_UniformBufferMemory[i]);
}

void HelloTriangleApplication::createDescriptorPool()
{
	std::array<VkDescriptorPoolSize, 2> PoolSizes = {};
	PoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	PoolSizes[0].descriptorCount = static_cast<uint32_t>(m_SwapChainImages.size());
	PoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	PoolSizes[1].descriptorCount = static_cast<uint32_t>(m_SwapChainImages.size());
	
	VkDescriptorPoolCreateInfo PoolInfo = {};
	PoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	PoolInfo.poolSizeCount = static_cast<uint32_t>(PoolSizes.size());
	PoolInfo.pPoolSizes = PoolSizes.data();
	PoolInfo.maxSets = static_cast<uint32_t>(m_SwapChainImages.size());
	if (!IS_VK_SUCCESS(vkCreateDescriptorPool(m_Device, &PoolInfo, nullptr, &m_DescriptorPool)))
		VERBOSE_EXIT("failed to create descriptor pool");
}

void HelloTriangleApplication::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> Layouts(m_SwapChainImages.size(), m_DescriptorSetLayout);
	VkDescriptorSetAllocateInfo AllocInfo = {};
	AllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	AllocInfo.descriptorPool = m_DescriptorPool;
	AllocInfo.descriptorSetCount = static_cast<uint32_t>(m_SwapChainImages.size());
	AllocInfo.pSetLayouts = Layouts.data();

	VkDescriptorPool DescriptorPool;
	m_DescriptorSets.resize(m_SwapChainImages.size());
	if (!IS_VK_SUCCESS(vkAllocateDescriptorSets(m_Device, &AllocInfo, m_DescriptorSets.data())))
		VERBOSE_EXIT("failed to allocate descriptor sets");

	for (size_t i = 0; i < m_SwapChainImages.size(); i++)
	{
		VkDescriptorBufferInfo BufferInfo = {};
		BufferInfo.buffer = m_UniformBuffers[i];
		BufferInfo.offset = 0;
		BufferInfo.range = sizeof(UniformBufferObject);

		VkDescriptorImageInfo ImageInfo = {};
		ImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		ImageInfo.imageView = m_TextureImageView;
		ImageInfo.sampler = m_TextureSampler;
		
		std::array<VkWriteDescriptorSet, 2> DescriptorWrites = {};
		DescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		DescriptorWrites[0].dstSet = m_DescriptorSets[i];
		DescriptorWrites[0].dstBinding = 0;
		DescriptorWrites[0].dstArrayElement = 0;
		DescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		DescriptorWrites[0].descriptorCount = 1;
		DescriptorWrites[0].pBufferInfo = &BufferInfo;

		DescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		DescriptorWrites[1].dstSet = m_DescriptorSets[i];
		DescriptorWrites[1].dstBinding = 1;
		DescriptorWrites[1].dstArrayElement = 0;
		DescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		DescriptorWrites[1].descriptorCount = 1;
		DescriptorWrites[1].pImageInfo = &ImageInfo;
		
		vkUpdateDescriptorSets(m_Device, DescriptorWrites.size(), DescriptorWrites.data(), 0, nullptr);
	}
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

		std::array<VkClearValue, 2> ClearVaules = {};
		ClearVaules[0].color = { 0.0f, 0.0f,0.0f,1.0f };
		ClearVaules[1].depthStencil = { 1.0f,0 };
		VkRenderPassBeginInfo RenderPassBeginInfo = {};
		RenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		RenderPassBeginInfo.renderPass = m_RenderPass;
		RenderPassBeginInfo.framebuffer = m_SwapChainFrameBuffers[i];
		RenderPassBeginInfo.renderArea.offset = { 0, 0 };
		RenderPassBeginInfo.renderArea.extent = m_SwapChainExtent;
		RenderPassBeginInfo.clearValueCount = static_cast<uint32_t>(ClearVaules.size());
		RenderPassBeginInfo.pClearValues = ClearVaules.data();

		vkCmdBeginRenderPass(m_CommandBuffers[i], &RenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(m_CommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_GraphicsPipeline);
		VkBuffer VertexBuffers[] = { m_VertexBuffer };
		VkDeviceSize Offsets[] = { 0 };
		vkCmdBindVertexBuffers(m_CommandBuffers[i], 0, 1, VertexBuffers, Offsets);
		vkCmdBindIndexBuffer(m_CommandBuffers[i], m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindDescriptorSets(m_CommandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSets[i], 0, nullptr);
		vkCmdDrawIndexed(m_CommandBuffers[i], static_cast<uint32_t>(m_Indices.size()), 1, 0, 0, 0);
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
	VkResult Result = vkAcquireNextImageKHR(m_Device, m_SwapChain, UINT64_MAX, m_ImageAvailableSemaphores[m_CurrentFrame], VK_NULL_HANDLE, &ImageIndex);
	if (Result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreateSwapChain();
		return;
	}
	else if (!IS_VK_SUCCESS(Result) && Result != VK_SUBOPTIMAL_KHR)
	{
		VERBOSE_EXIT("failed to acquire swap chain image");
	}

	updateUniformBuffer(ImageIndex);
	
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

	vkResetFences(m_Device, 1, &m_InFlightFences[m_CurrentFrame]);
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
	Result = vkQueuePresentKHR(m_PresentQueue, &PresentInfo);
	if (Result == VK_ERROR_OUT_OF_DATE_KHR || Result == VK_SUBOPTIMAL_KHR)
	{
		m_FramebufferResized = false;
		recreateSwapChain();
	}
	else if (Result != VK_SUCCESS)
	{
		VERBOSE_EXIT("failed to present swap chain image");
	}

	m_CurrentFrame = (m_CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void HelloTriangleApplication::updateUniformBuffer(uint32_t vCurrentImage)
{
	static auto StartTime = std::chrono::high_resolution_clock::now();
	auto CurrentTime = std::chrono::high_resolution_clock::now();
	float Time = std::chrono::duration<float, std::chrono::seconds::period>(CurrentTime - StartTime).count();

	UniformBufferObject Ubo = {};
	Ubo.Model = glm::rotate(glm::mat4(1.0f), Time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Ubo.View = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Ubo.Proj = glm::perspective(glm::radians(45.0f), m_SwapChainExtent.width / (float)m_SwapChainExtent.height, 0.1f, 10.0f);
	Ubo.Proj[1][1] *= -1;

	void* Data;
	vkMapMemory(m_Device, m_UniformBufferMemory[vCurrentImage], 0, sizeof(Ubo), 0, &Data);
	memcpy(Data, &Ubo, sizeof(Ubo));
	vkUnmapMemory(m_Device, m_UniformBufferMemory[vCurrentImage]);
}

void HelloTriangleApplication::recreateSwapChain()
{
	int Width = 0, Height = 0;
	while(Width == 0 || Height == 0)
	{
		glfwGetFramebufferSize(m_pWindow, &Width, &Height);
		glfwWaitEvents();
	}
	vkDeviceWaitIdle(m_Device);

	cleanupSwapChain();
	
	createSwapChain();
	createImageViews();
	createRenderPass();
	createGraphicsPipeline();
	createColorResources();
	createDepthResources();
	createFrameBuffers();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffers();
}

void HelloTriangleApplication::cleanupSwapChain()
{
	vkDestroyImageView(m_Device, m_ColorImageView, nullptr);
	vkDestroyImage(m_Device, m_ColorImage, nullptr);
	vkFreeMemory(m_Device, m_ColorImageMemory, nullptr);
	
	vkDestroyImageView(m_Device, m_DepthImageView, nullptr);
	vkDestroyImage(m_Device, m_DepthImage, nullptr);
	vkFreeMemory(m_Device, m_DepthImageMemory, nullptr);
	
	for (size_t i = 0; i < m_SwapChainFrameBuffers.size(); i++)
		vkDestroyFramebuffer(m_Device, m_SwapChainFrameBuffers[i], nullptr);
	vkFreeCommandBuffers(m_Device, m_CommandPool, static_cast<uint32_t>(m_CommandBuffers.size()), m_CommandBuffers.data());
	vkDestroyPipeline(m_Device, m_GraphicsPipeline, nullptr);
	vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
	vkDestroyRenderPass(m_Device, m_RenderPass, nullptr);

	for (size_t i = 0; i < m_SwapChainImageViews.size(); i++)
		vkDestroyImageView(m_Device, m_SwapChainImageViews[i], nullptr);
	vkDestroySwapchainKHR(m_Device, m_SwapChain, nullptr);

	for (size_t i = 0; i < m_SwapChainImages.size(); i++)
	{
		vkDestroyBuffer(m_Device, m_UniformBuffers[i], nullptr);
		vkFreeMemory(m_Device, m_UniformBufferMemory[i], nullptr);
	}

	vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
}

void HelloTriangleApplication::cleanup()
{
	cleanupSwapChain();

	vkDestroySampler(m_Device, m_TextureSampler, nullptr);
	vkDestroyImageView(m_Device, m_TextureImageView, nullptr);
	
	vkDestroyImage(m_Device, m_TextureImage, nullptr);
	vkFreeMemory(m_Device, m_TextureImageMemory, nullptr);
	
	vkDestroyDescriptorSetLayout(m_Device, m_DescriptorSetLayout, nullptr);
	
	vkDestroyBuffer(m_Device, m_VertexBuffer, nullptr);
	vkFreeMemory(m_Device, m_VertexBufferMemory, nullptr);
	vkDestroyBuffer(m_Device, m_IndexBuffer, nullptr);
	vkFreeMemory(m_Device, m_IndexBufferMemory, nullptr);
	
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
	{
		vkDestroySemaphore(m_Device, m_RenderFinishedSemaphores[i], nullptr);
		vkDestroySemaphore(m_Device, m_ImageAvailableSemaphores[i], nullptr);
		vkDestroyFence(m_Device, m_InFlightFences[i], nullptr);
	}
	vkDestroyCommandPool(m_Device, m_CommandPool, nullptr);
	
	vkDestroyDevice(m_Device, nullptr);
	if constexpr (EnableValidationLayers) DestroyDebugUtilsMessengerEXT(m_Instance, m_DebugMessenger, nullptr);
	vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
	vkDestroyInstance(m_Instance, nullptr);
	
	glfwDestroyWindow(m_pWindow);
	glfwTerminate();
}
