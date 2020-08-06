#include "Igniter.h"
#include "SpinningCube.h"
#include "SpinningTexturePlane.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	CSpinningCube SpinningCube;
	CSpinningTexturePlane SpinningTexturePlane;
	
	CIgniter::start(hInstance);
	CIgniter::run(&SpinningTexturePlane);
	CIgniter::shutdown();
	
	return 0;
}