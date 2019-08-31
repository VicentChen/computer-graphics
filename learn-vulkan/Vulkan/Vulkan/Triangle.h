#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

// ----- macros for self-implemented logs ----- //
#define USE_VERBOSE_EXIT 1
#define CREATE_INSTANCE_VERBOSE 0
#define CHECK_VALIDATION_LAYER_SUPPORT_VERBOSE 0
#define CHECK_DEVICE_EXTENSION_SUPPORT_VERBOSE 0
#define PICK_PHYSICAL_DEVICE_VERBOSE 1

#define OUTPUT_HEADER() \
	do { \
		std::cout << "----- " << __FILE__ << ' ' << __LINE__ << ": " << __func__ << " -----" << std::endl;\
	} while(0)

#if USE_VERBOSE_EXIT
#define VERBOSE_EXIT(message) do {\
		std::cerr << "===== PROGRAM STOP =====" << std::endl; \
		std::cout << __FILE__ << ' ' << __LINE__ << ": " << __func__ << std::endl; \
		std::cerr << "[FATAL]: " << (message) << std::endl; \
		exit(-1); \
	}while(0)
#else
#define VERBOSE_EXIT(message) do {\
		throw std::runtime_error(message); \
	}while(0)
#endif

#define IS_VK_SUCCESS(cond) ((cond) == VK_SUCCESS)

// ----- configurations ----- //
// GUI
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
// vulkan
#ifdef NDEBUG
const bool EnableValidationLayers = false;
#else
const bool EnableValidationLayers = true;
#endif
const int MAX_FRAMES_IN_FLIGHT = 2;
const std::vector<const char*> ValidationLayers = { "VK_LAYER_LUNARG_standard_validation" };
const std::vector<const char*> DeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

struct QueueFamilyIndices
{
	std::optional<uint32_t> GraphicsFamily;
	std::optional<uint32_t> PresentFamily;

	bool isComplete() { return GraphicsFamily.has_value() && PresentFamily.has_value(); }
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR Capabilities;
	std::vector<VkSurfaceFormatKHR> Formats;
	std::vector<VkPresentModeKHR> PresentModes;
};

class HelloTriangleApplication
{
public:
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT vMessageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT vMessageType,
		const VkDebugUtilsMessengerCallbackDataEXT* vCallbackData,
		void* vUserData);
	static std::vector<char> readFile(const std::string& vFileName);
	
	void run();
	
	void initWindow();

	void initVulkan();
	void createInstance();// initVulkan()
	bool checkValidationLayerSupport(); // createInstance()
	std::vector<const char*> getRequiredExtensions(); // createInstance()
	void setupDebugMessenger(); // initVulkan()
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& voCreateInfo); // setupDebugMessenger()
	void pickPhysicalDevice(); // initVulkan()
	bool isDeviceSuitable(VkPhysicalDevice vDevice); // pickPhysicalDevice()
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice vDevice); // isDeviceSuitable()
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice vDevice); // isDeviceSuitable()
	bool checkDeviceExtensionSupport(VkPhysicalDevice vDevice); // isDeviceSuitable()
	void createLogicalDevice(); // initVulkan()
	void createSurface(); // initVulkan()
	void createSwapChain(); // initVulkan()
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& vSupportedFormats); // createSwapChain()
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& vSupportedPresentModes); // createSwapChain()
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& vCapabilities); // createSwapChain()
	void createImageViews(); // initVulkan()
	void createRenderPass(); // initVulkan()
	void createGraphicsPipeline(); // initVulkan()
	VkShaderModule createShaderModule(const std::vector<char>& vCode); // createGraphicsPipeline()
	void createFrameBuffers(); // initVulkan()
	void createCommandPool(); // initVulkan()
	void createCommandBuffers(); // initVulkan()
	void createSyncObjects(); // initVulkan()
	
	void mainLoop();
	void drawFrame(); // mainLoop();
	
	void cleanup();
	
private:

	VkInstance m_Instance;
	VkDebugUtilsMessengerEXT m_DebugMessenger;
	VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;
	VkDevice m_Device;

	// ----- Queues ----- //
	VkQueue m_GraphicsQueue;
	VkQueue m_PresentQueue;

	VkSurfaceKHR m_Surface;

	// ----- Swap Chain ----- //
	VkSwapchainKHR m_SwapChain;
	std::vector<VkImage> m_SwapChainImages;
	VkFormat m_SwapChainImageFormat;
	VkExtent2D m_SwapChainExtent;
	std::vector<VkImageView> m_SwapChainImageViews;
	std::vector<VkFramebuffer> m_SwapChainFrameBuffers;

	// ----- Pipeline ----- //
	VkRenderPass m_RenderPass;
	VkPipelineLayout m_PipelineLayout;
	VkPipeline m_GraphicsPipeline;

	// ----- Commands ----- //
	VkCommandPool m_CommandPool;
	std::vector<VkCommandBuffer> m_CommandBuffers;

	// ----- Semaphores ----- //
	std::vector<VkSemaphore> m_ImageAvailableSemaphores;
	std::vector<VkSemaphore> m_RenderFinishedSemaphores;
	std::vector<VkFence> m_InFlightFences;
	size_t m_CurrentFrame = 0;
	
	// ----- GLFW window ----- //
	GLFWwindow* m_pWindow;
};

VkResult CreateDebugUtilsMessengerEXT(
	VkInstance vInstance,
	const VkDebugUtilsMessengerCreateInfoEXT* vCreateInfo,
	const VkAllocationCallbacks* vAllocator,
	VkDebugUtilsMessengerEXT* vDebugMessenger);

VkResult DestroyDebugUtilsMessengerEXT(
	VkInstance vInstance,
	VkDebugUtilsMessengerEXT vDebugMessenger,
	const VkAllocationCallbacks* vAllocator);