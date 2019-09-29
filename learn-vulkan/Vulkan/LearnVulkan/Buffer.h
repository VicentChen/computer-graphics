#pragma once
#include "Config.h"
#include "Device.h"

namespace LearnVulkan
{
	class Buffer
	{
	public:
		Buffer(vk::UniqueBuffer&& vBuffer, vk::UniqueDeviceMemory&& vMemory, const void* vData, uint32_t vSize, Device* vDevice) : m_Buffer(std::move(vBuffer)), m_Memory(std::move(vMemory)), m_pDevice(vDevice), m_pData(vData), m_Size(vSize) { }
		vk::Buffer fetchBuffer() { return m_Buffer.get(); }
		vk::DeviceMemory fetchMemory() { return m_Memory.get(); }

		const void* ptr() const { return m_pData; }
		uint32_t size() const { return m_Size; }
		
	private:
		vk::UniqueBuffer m_Buffer;
		vk::UniqueDeviceMemory m_Memory;
		Device* m_pDevice = nullptr;
		const void* m_pData = nullptr;
		uint32_t m_Size = 0;

		friend class Device;
	};
}