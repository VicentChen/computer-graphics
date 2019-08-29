#include "Triangle.h"
#include <algorithm>

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
	if (EnableValidationLayers)
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
	std::cout << "Required Extensions: " << std::endl;
	for (const char* pRequiredExtension : RequiredExtensions)
	{
		const VkExtensionProperties* pExtension = reinterpret_cast<const VkExtensionProperties*>(pRequiredExtension);
		std::cout << '\t' << pExtension->extensionName << std::endl;
	}
#endif

#if CREATE_INSTANCE_VERBOSE
	// ----- find all extensions that vulkan supports ----- //
	uint32_t SupportedExtensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &SupportedExtensionCount, nullptr);
	std::vector<VkExtensionProperties> SupportedExtensions(SupportedExtensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &SupportedExtensionCount, SupportedExtensions.data());
	
	OUTPUT_HEADER();
	std::cout << "Supported Extensions: " << std::endl;
	for (const auto& Extension : SupportedExtensions)
		std::cout << "\t" << Extension.extensionName << std::endl;
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
	if (EnableValidationLayers) Extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	return Extensions;
}

void HelloTriangleApplication::setupDebugMessenger()
{
	if (!EnableValidationLayers) return;
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

void HelloTriangleApplication::mainLoop()
{
	while (!glfwWindowShouldClose(m_pWindow))
	{
		glfwPollEvents();
	}
}

void HelloTriangleApplication::cleanup()
{
	if (EnableValidationLayers) DestroyDebugUtilsMessengerEXT(m_Instance, m_DebugMessenger, nullptr);
	
	vkDestroyInstance(m_Instance, nullptr);
	
	glfwDestroyWindow(m_pWindow);
	glfwTerminate();
}
