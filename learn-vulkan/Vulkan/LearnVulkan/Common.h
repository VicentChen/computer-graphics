#pragma once
#include "Log.h"
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define CAST_U32I(i) static_cast<uint32_t>(i)
#define CAST_FLT(i) static_cast<float>(i)

#define ENABLE_CHECK 1

namespace LearnVulkan
{
	inline bool isVkSuccess(vk::Result vResult) { return vResult == vk::Result::eSuccess; }

	inline VkBool32 debugReportCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT /*objectType*/, uint64_t /*object*/, size_t /*location*/, int32_t /*messageCode*/, const char* /*pLayerPrefix*/, const char* pMessage, void* /*pUserData*/)
	{
		switch (flags)
		{
		case VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
			std::cerr << "INFORMATION: ";
			break;
		case VK_DEBUG_REPORT_WARNING_BIT_EXT:
			std::cerr << "WARNING: ";
			break;
		case VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
			std::cerr << "PERFORMANCE WARNING: ";
			break;
		case VK_DEBUG_REPORT_ERROR_BIT_EXT:
			std::cerr << "ERROR: ";
			break;
		case VK_DEBUG_REPORT_DEBUG_BIT_EXT:
			std::cerr << "DEBUG: ";
			break;
		default:
			std::cerr << "unknown flag (" << flags << "): ";
			break;
		}
		std::cerr << pMessage << std::endl;
		return VK_TRUE;
	}

	inline VkBool32 debugMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT vMessageSeverity, VkDebugUtilsMessageTypeFlagsEXT vMessageType, const VkDebugUtilsMessengerCallbackDataEXT* vCallbackData, void* vUserData) {
		switch (vMessageType)
		{
		case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:     std::cerr << "[  GENERAL  ]"; break;
		case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:  std::cerr << "[VALIDATION ]"; break;
		case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: std::cerr << "[PERFORMANCE]"; break;
		default:                                              std::cerr << "[  UNKNOWN  ]"; break;
		}

		switch (vMessageSeverity)
		{
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: std::cerr << "[VERBOSE]"; break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:    std::cerr << "[ INFO  ]"; break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: std::cerr << "[WARNING]"; break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:   std::cerr << "[ ERROR ]"; break;
		default:                                              std::cerr << "[UNKNOWN]"; break;
		}
		std::cerr << ": ";
		std::cerr << vCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	inline bool checkInstanceExtensionSupport(vk::Instance& vInstance, const std::vector<const char*>& vLayers, const std::vector<const char*>& vExtensions)
	{
#if ENABLE_CHECK
		std::set<std::string> RequestedLayerSet(vLayers.begin(), vLayers.end());
		std::set<std::string> RequestedExtensionSet(vExtensions.begin(), vExtensions.end());

		auto SupportedLayers = vk::enumerateInstanceLayerProperties();
		for (const auto& SupportedLayer : SupportedLayers) RequestedLayerSet.erase(SupportedLayer.layerName);
		if (!RequestedLayerSet.empty())
		{
			LOG_ERROR("Instance layers not supported: ");
			for (const auto& RequestedLayer : RequestedLayerSet)
				LOG_ERROR("\t" + RequestedLayer);
		}

		for (const auto& SupportedExtension : vk::enumerateInstanceExtensionProperties(nullptr))
			RequestedExtensionSet.erase(SupportedExtension.extensionName);
		for (const auto& pLayerName : vLayers)
		{
			for (const auto& SupportedExtension : vk::enumerateInstanceExtensionProperties(std::string(pLayerName)))
				RequestedExtensionSet.erase(SupportedExtension.extensionName);
		}
		if (!RequestedExtensionSet.empty())
		{
			LOG_ERROR("Instance extension not supported: ");
			for (const auto& RequestedExtension : RequestedExtensionSet)
				LOG_ERROR("\t" + RequestedExtension);
		}
#endif
		return true;
	}

	inline bool checkPhysicalDeviceSupport(vk::PhysicalDevice& vPhysicalDevice, vk::QueueFlags vFlags, const std::vector<const char*>& vExtensions)
	{
#if ENABLE_CHECK
		auto QueueFamilies = vPhysicalDevice.getQueueFamilyProperties();
#endif		
		return true;
	}

	inline std::vector<char> readFile(const std::string& vFilePath)
	{
		std::ifstream File(vFilePath, std::ios::ate | std::ios::binary);

		if (!File.is_open()) VERBOSE_EXIT("File not opened");

		size_t FileSize = static_cast<size_t>(File.tellg());
		std::vector<char> Buffer(FileSize);

		File.seekg(0);
		File.read(Buffer.data(), FileSize);

		File.close();

		return Buffer;
	}
}