#include "Triangle.h"

int main(int argc, char* argv[])
{
	HelloTriangleApplication App;

	try
	{
		App.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
