#include "Igniter.h"
#include "SpinningCube.h"
#include "SpinningTexturePlane.h"
#include "ShadowTexturePlane.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	CSpinningCube SpinningCube;
	CSpinningTexturePlane SpinningTexturePlane;
	CShadowTexturePlane ShadowTexturePlane;
	
	CIgniter::start(hInstance);
	CIgniter::run(&ShadowTexturePlane);
	CIgniter::shutdown();
	
	return 0;
}