#pragma once
#include "Device.h"

namespace LearnVulkan
{
	class Queue
	{
	public:
		const std::string& getName() const { return m_Name; }
		uint32_t getIndex() const { return m_Index; }
		vk::Queue& fetchQueue() { return m_Queue; }
		static Queue createdByDevice(Device& vDevice, const std::string& vName, uint32_t vCount = 1) { return vDevice.initQueue(vName, vCount); }
		
	private:
		Queue(const std::string& vName, const vk::Queue& vQueue, uint32_t vIndex) : m_Name(vName), m_Queue(vQueue), m_Index(vIndex) {}
		
		const std::string m_Name;
		vk::Queue m_Queue;
		uint32_t m_Index;

		friend class Device;
	};
}