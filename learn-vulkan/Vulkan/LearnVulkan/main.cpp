#include "Config.h"
#include "Instance.h"
#include "Device.h"

using namespace LearnVulkan;

int main(int argc, char* argv[])
{
	//Window window;
	//window.display();
	Instance Instance(Default::Application::Info);
	PhysicalDevice PhysicalDevice = Instance.initPhysicalDevice();
	Device Device = PhysicalDevice.initDevice();
	return 0;
}
