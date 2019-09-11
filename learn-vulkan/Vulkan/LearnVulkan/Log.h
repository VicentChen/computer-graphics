#pragma once
#include <string>

#define USE_VERBOSE_EXIT 1

#if USE_VERBOSE_EXIT
#define VERBOSE_EXIT(message) do {\
		std::cerr << "===== PROGRAM STOP =====" << std::endl; \
		std::cout << __FILE__ << ' ' << __LINE__ << ": " << __func__ << std::endl; \
		std::cerr << "[FATAL]: " << (message) << std::endl; \
		exit(-1); \
	}while(0)
#else
#define VERBOSE_EXIT(message) do {\
		throw std::runtime_error(message); \
	}while(0)
#endif

// TODO: extrace filename
#define LOG_HEADER() \
	do { \
		std::cout << "----- " << __FILE__ << ' ' << __LINE__ << ": " << __func__ << " -----" << std::endl;\
	} while(0)

#define LOG_INFO(Msg) \
	do { \
		LOG_HEADER(); \
		std::cout << "[INFO ] " << (Msg) << std::endl; \
	} while(0)

#define LOG_ERROR(Msg) \
	do {\
		LOG_HEADER(); \
		std::cout << "[ERROR] " << (Msg) << std::endl; \
	} while(0)