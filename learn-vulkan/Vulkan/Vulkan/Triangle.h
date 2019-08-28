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

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}
	
	void initWindow();
	
	void initVulkan();
	void createInstance();
	bool checkValidationLayerSupport();
	
	void mainLoop();
	void cleanup();
	
private:

	VkInstance m_Instance;
	
	// ----- GLFW window ----- //
	GLFWwindow* m_pWindow;
};