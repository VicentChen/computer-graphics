#include "Triangle.h"
#include <algorithm>

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

	uint32_t GLFWExtensionCount = 0;
	const char** pGLFWExtensions;
	pGLFWExtensions = glfwGetRequiredInstanceExtensions(&GLFWExtensionCount);
	VkInstanceCreateInfo CreateInfo = {};
	CreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	CreateInfo.pApplicationInfo = &AppInfo;
	CreateInfo.enabledExtensionCount = GLFWExtensionCount;
	CreateInfo.ppEnabledExtensionNames = pGLFWExtensions;
	CreateInfo.enabledLayerCount = 0;
	if (EnableValidationLayers)
	{
		CreateInfo.enabledLayerCount = static_cast<uint32_t>(ValidationLayers.size());
		CreateInfo.ppEnabledLayerNames = ValidationLayers.data();
	}

#if CREATE_INSTANCE_VERBOSE
	OUTPUT_HEADER();
	std::cout << "GLFW Required Extensions: " << std::endl;
	const VkExtensionProperties** pGLFWExtensionProperties = reinterpret_cast<const VkExtensionProperties**>(pGLFWExtensions);
	for (int i = 0; i < GLFWExtensionCount; i++)
		std::cout << '\t' << pGLFWExtensionProperties[i]->extensionName << std::endl;
#endif

#if CREATE_INSTANCE_VERBOSE
	// ----- find all extensions that vulkan supports ----- //
	uint32_t ExtensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &ExtensionCount, nullptr);
	std::vector<VkExtensionProperties> Extensions(ExtensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &ExtensionCount, Extensions.data());
	
	OUTPUT_HEADER();
	std::cout << "Available Extensions: " << std::endl;
	for (const auto& Extension : Extensions)
		std::cout << "\t" << Extension.extensionName << std::endl;
#endif

	// ----- create instance ----- //
	if (vkCreateInstance(&CreateInfo, nullptr, &m_Instance) != VK_SUCCESS)
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

void HelloTriangleApplication::mainLoop()
{
	while (!glfwWindowShouldClose(m_pWindow))
	{
		glfwPollEvents();
	}
}

void HelloTriangleApplication::cleanup()
{
	vkDestroyInstance(m_Instance, nullptr);
	
	glfwDestroyWindow(m_pWindow);
	glfwTerminate();
}
