#include "Igniter.h"
#include "Application.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	CApplication App;
	
	CIgniter::start(hInstance);
	CIgniter::run(&App);
	CIgniter::shutdown();
	
	return 0;
}