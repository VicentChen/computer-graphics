#pragma once
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <GLFW/glfw3.h>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
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
// model
const std::string MODEL_PATH = "Models/chalet.obj";
const std::string TEXTURE_PATH = "Textures/chalet.jpg";
// vulkan
#ifdef NDEBUG
const bool EnableValidationLayers = false;
#else
const bool EnableValidationLayers = true;
#endif
const int MAX_FRAMES_IN_FLIGHT = 2;
const std::vector<const char*> ValidationLayers = { "VK_LAYER_LUNARG_standard_validation" };
const std::vector<const char*> DeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

struct Vertex
{
	glm::vec3 Pos;
	glm::vec3 Color;
	glm::vec2 TexCoord;

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription BindingDescription = {};
		BindingDescription.binding = 0;
		BindingDescription.stride = sizeof(Vertex);
		BindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return BindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> AttributeDescriptions = {};
		AttributeDescriptions[0].binding = 0;
		AttributeDescriptions[0].location = 0;
		AttributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		AttributeDescriptions[0].offset = offsetof(Vertex, Pos);
		AttributeDescriptions[1].binding = 0;
		AttributeDescriptions[1].location = 1;
		AttributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		AttributeDescriptions[1].offset = offsetof(Vertex, Color);
		AttributeDescriptions[2].binding = 0;
		AttributeDescriptions[2].location = 2;
		AttributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		AttributeDescriptions[2].offset = offsetof(Vertex, TexCoord);
		return AttributeDescriptions;
	}

	bool operator==(const Vertex& vOther) const
	{
		return Pos == vOther.Pos && Color == vOther.Color && TexCoord == vOther.TexCoord;
	}
};

namespace std
{
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vVertex) const
		{
			return ((hash<glm::vec3>()(vVertex.Pos) ^
				(hash<glm::vec3>()(vVertex.Color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vVertex.TexCoord) << 1);
		}
	};
}

struct UniformBufferObject
{
	glm::mat4 Model;
	glm::mat4 View;
	glm::mat4 Proj;
};

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
	static void framebufferResizeCallback(GLFWwindow* vWindow, int vWidth, int vHeight);
	
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
	VkSampleCountFlagBits getMaxUsableSampleCount(); // pickPhysicalDevice()
	void createLogicalDevice(); // initVulkan()
	void createSurface(); // initVulkan()
	void createSwapChain(); // initVulkan()
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& vSupportedFormats); // createSwapChain()
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& vSupportedPresentModes); // createSwapChain()
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& vCapabilities); // createSwapChain()
	void createImageViews(); // initVulkan()
	void createRenderPass(); // initVulkan()
	void createDescriptorSetLayout(); // initVulkan()
	void createGraphicsPipeline(); // initVulkan()
	VkShaderModule createShaderModule(const std::vector<char>& vCode); // createGraphicsPipeline()
	void createFrameBuffers(); // initVulkan()
	void createCommandPool(); // initVulkan()
	void createColorResources(); // initVulkan()
	void createDepthResources(); // initVulkan()
	VkFormat findDepthFormat(); // createDepthResources()
	bool hasStencilComponent(VkFormat vFormat);
	VkFormat findSupportedFormat(const std::vector<VkFormat>& vCandidates, VkImageTiling vTiling, VkFormatFeatureFlags vFeatures);
	void loadModel(); // initVulkan()
	void createVertexBuffer(); // initVulkan()
	uint32_t findMemoryType(uint32_t vTypeFilter, VkMemoryPropertyFlags vProperties); // createVertexBuffer()
	void createBuffer(VkDeviceSize vSize, VkBufferUsageFlags vUsage, VkMemoryPropertyFlags vProperties, VkBuffer& vBuffer, VkDeviceMemory& vBufferMemory); // createVertexBuffer()
	void copyBuffer(VkBuffer vSrcBuffer, VkBuffer vDstBuffer, VkDeviceSize vSize); // createVertexBuffer()
	void createIndexBuffer(); // initVulkan()
	void createUniformBuffers(); // initVulkan()
	void createCommandBuffers(); // initVulkan()
	void createTextureImage(); // initVulkan()
	void generateMipmaps(VkImage vImage, VkFormat vImageFormat, int32_t vTexWidth, int32_t vTexHeight, uint32_t vMipLevels);
	void createTextureImageView(); // initVulkan()
	void createTextureSampler(); // initVulkan()
	VkImageView createImageView(VkImage vImage, VkFormat vFormat, VkImageAspectFlags vAspectFlags, uint32_t vMipLevels);
	void copyBufferToImage(VkBuffer vBuffer, VkImage vImage, uint32_t vWidth, uint32_t vHeight); // createTextureImage()
	void createImage(uint32_t vWidth, uint32_t vHeight, uint32_t vMipLevels, VkSampleCountFlagBits vNumSamples, VkFormat vFormat, VkImageTiling vTiling, VkImageUsageFlags vUsage, VkMemoryPropertyFlags vProperties, VkImage& vImage, VkDeviceMemory& vImageMemory); // createTextureImage()
	VkCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(VkCommandBuffer vCommandBuffer);
	void transitionImageLayout(VkImage vImage, VkFormat vFormat, VkImageLayout vOldLayout, VkImageLayout vNewLayout, uint32_t vMipLevels);
	void createDescriptorPool(); // initVulkan()
	void createDescriptorSets(); // initVulkan()
	void createSyncObjects(); // initVulkan()
	
	void mainLoop();
	void drawFrame(); // mainLoop()
	void updateUniformBuffer(uint32_t vCurrentImage);
	void recreateSwapChain();
	void cleanupSwapChain(); // recreateSwapChain()
	
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
	VkDescriptorSetLayout m_DescriptorSetLayout;
	std::vector<VkDescriptorSet> m_DescriptorSets;
	VkDescriptorPool m_DescriptorPool;
	VkPipelineLayout m_PipelineLayout;
	VkPipeline m_GraphicsPipeline;

	std::vector<Vertex> m_Vertices;
	std::vector<uint32_t> m_Indices;
	VkBuffer m_VertexBuffer;
	VkDeviceMemory m_VertexBufferMemory;
	VkBuffer m_IndexBuffer;
	VkDeviceMemory m_IndexBufferMemory;
	std::vector<VkBuffer> m_UniformBuffers;
	std::vector<VkDeviceMemory> m_UniformBufferMemory;

	// ----- Multi Sample ----- //
	VkSampleCountFlagBits m_MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	VkImage m_ColorImage;
	VkDeviceMemory m_ColorImageMemory;
	VkImageView m_ColorImageView;
	
	// ----- Textures ----- //
	uint32_t m_MipLevels;
	VkImage m_TextureImage;
	VkDeviceMemory m_TextureImageMemory;
	VkImageView m_TextureImageView;
	VkSampler m_TextureSampler;

	// ----- Depth ----- //
	VkImage m_DepthImage;
	VkDeviceMemory m_DepthImageMemory;
	VkImageView m_DepthImageView;
	
	// ----- Commands ----- //
	VkCommandPool m_CommandPool;
	std::vector<VkCommandBuffer> m_CommandBuffers;

	// ----- Semaphores ----- //
	std::vector<VkSemaphore> m_ImageAvailableSemaphores;
	std::vector<VkSemaphore> m_RenderFinishedSemaphores;
	std::vector<VkFence> m_InFlightFences;
	size_t m_CurrentFrame = 0;
	bool m_FramebufferResized = false;
	
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