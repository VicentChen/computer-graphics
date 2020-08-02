#include "Igniter.h"
#include "SpinningCube.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	CSpinningCube App;
	
	CIgniter::start(hInstance);
	CIgniter::run(&App);
	CIgniter::shutdown();
	
	return 0;
}