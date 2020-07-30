#pragma once
#include "Common.h"

class CApplication
{
public:

	CApplication() = default;
	virtual ~CApplication() = default;

	virtual void start() {}
	virtual void update() {}
	virtual void render();
	virtual void shutdown() {}

	virtual void onKey(WPARAM vParam) {}
};
