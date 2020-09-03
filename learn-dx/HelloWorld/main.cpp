#include "Igniter.h"
#include "SpinningCube.h"
#include "SpinningTexturePlane.h"
#include "ShadowTexturePlane.h"
#include "RayTracingTriangle.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
	CSpinningCube SpinningCube;
	CSpinningTexturePlane SpinningTexturePlane;
	CShadowTexturePlane ShadowTexturePlane;
	CRayTracingTriangle RayTracingTriangle;
	
	CIgniter::start(hInstance);
	CIgniter::run(&RayTracingTriangle);
	CIgniter::shutdown();
	
	return 0;
}