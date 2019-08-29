#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

// ----- macros for self-implemented logs ----- //
#define USE_VERBOSE_EXIT 1
#define CREATE_INSTANCE_VERBOSE 1
#define CHECK_VALIDATION_LAYER_SUPPORT_VERBOSE 1

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
const std::vector<const char*> ValidationLayers = { "VK_LAYER_LUNARG_standard_validation" };

VkResult CreateDebugUtilsMessengerEXT(
	VkInstance vInstance,
	const VkDebugUtilsMessengerCreateInfoEXT* vCreateInfo,
	const VkAllocationCallbacks* vAllocator,
	VkDebugUtilsMessengerEXT* vDebugMessenger);

VkResult DestroyDebugUtilsMessengerEXT(
	VkInstance vInstance,
	VkDebugUtilsMessengerEXT vDebugMessenger,
	const VkAllocationCallbacks* vAllocator);

class HelloTriangleApplication
{
public:
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT vMessageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT vMessageType,
		const VkDebugUtilsMessengerCallbackDataEXT* vCallbackData,
		void* vUserData);
	
	void run();
	
	void initWindow();

	void initVulkan();
	void createInstance();// initVulkan()
	bool checkValidationLayerSupport(); // createInstance()
	std::vector<const char*> getRequiredExtensions(); // createInstance()
	void setupDebugMessenger(); // initVulkan()
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& voCreateInfo); // setupDebugMessenger()
	
	void mainLoop();
	void cleanup();
	
private:

	VkInstance m_Instance;
	VkDebugUtilsMessengerEXT m_DebugMessenger;
	
	// ----- GLFW window ----- //
	GLFWwindow* m_pWindow;
};