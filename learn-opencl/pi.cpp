// OpenCl-1.cpp : Defines the entry point for the console application.
//
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

using namespace std;

// 将cl文件代码转为字符串
bool convertCLFile2String(const char *pFileName, std::string& Str);

int main(int argc, char* argv[])
{
  srand((unsigned)time(NULL));

  cl_int err;
	// 要求和的int数组
	size_t data_size = 100000000;
	int *data_in = new int[data_size];//1.5亿元素的数组
	for (size_t i = 0; i < data_size; i += 2) {
    data_in[i] = rand() % 250 - 125; // x
	  data_in[i + 1] = rand() % 250 - 125; // y
	}

	cl_int iStatus = 0;	// 函数返回状态



	// -------------------1. 获得并选择可用平台-----------------------------
	cl_uint	uiNumPlatforms = 0;		// 平台个数
	// 查询可用的平台个数，并返回状态
	iStatus = clGetPlatformIDs(0, NULL, &uiNumPlatforms);
	if (CL_SUCCESS != iStatus)
	{
		cout << "Error: Getting platforms error" << endl;
		getchar();
		return 0;
	}

	cl_platform_id	Platform = NULL;	// 选择的平台
	// 获得平台地址
	if (uiNumPlatforms > 0)  // 如果有可用平台
	{
		// 根据平台数为平台分配内存空间
		cl_platform_id *pPlatforms = new cl_platform_id[uiNumPlatforms];

		// 获得可用的平台
		iStatus = clGetPlatformIDs(uiNumPlatforms, pPlatforms, NULL);
		Platform = pPlatforms[1];		// 获得第一个平台的地址
		delete[] pPlatforms;			// 释放平台占用的内存空间
	}

	// 获得平台版本名
	size_t	uiSize = 0;	// 平台版本名字字节数	
	// 获得平台版本名的字节数
	iStatus = clGetPlatformInfo(Platform, CL_PLATFORM_VERSION, 0, NULL, &uiSize);
	// 根据字节数为平台版本名分配内存空间
	char *pName = new char[uiSize];
	// 获得平台版本名字
	iStatus = clGetPlatformInfo(Platform, CL_PLATFORM_VERSION, uiSize, pName, NULL);
	cout << pName << endl;
	delete[] pName;



	//--------------2. 查询GPU设备，并选择可用设备------------------------
	cl_uint	uiNumDevices = 0;	// 设备数量
								// 获得GPU设备数量
	iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
	cl_device_id	*pDevices = NULL;	// 设备
	if (0 == uiNumDevices)				// 如果没有GPU设备
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;

		// 选择CPU作为设备，获得设备数
		iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);

		// 为设备分配空间
		pDevices = new cl_device_id[uiNumDevices];

		// 获得平台
		iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_CPU, uiNumDevices, pDevices, NULL);
	}
	else
	{
		// 为设备分配空间
		pDevices = new cl_device_id[uiNumDevices];
		iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_GPU, uiNumDevices, pDevices, NULL);
	}



	// -------------------3.创建设备上下文---------------------------------
	cl_context Context = NULL;	//设备上下文
								// 创建设备上下文
	Context = clCreateContext(NULL, 1, pDevices, NULL, NULL, NULL);
	if (NULL == Context)
	{
		cout << "Error: Can not create context" << endl;
		getchar();
		return 0;
	}



	// -------------------4.创建命令队列--------------------------------------
	cl_command_queue CommandQueue = NULL;
	// 创建第1个设备的命令队列
	CommandQueue = clCreateCommandQueue(Context, pDevices[0], 0, NULL);
	if (NULL == CommandQueue)
	{
		cout << "Error: Can not create CommandQueue" << endl;
		getchar();
		return 0;
	}



	// ----------------------5. 创建程序对象------------------------------
	const char	*pFileName = "pi.cl";	// cl文件名
	string	strSource = "";		// 用于存储cl文件中的代码
								// 将cl文件中的代码转为字符串
	iStatus = convertCLFile2String(pFileName, strSource);

	const char *pSource = strSource.c_str();// 获得strSource指针
	size_t uiArrSourceSize[] = { strlen(pSource) };	// 字符串大小

	// 创建程序对象
	cl_program Program = clCreateProgramWithSource(Context, 1, &pSource, uiArrSourceSize, NULL);
	if (NULL == Program)
	{
		cout << "Error: Can not create program" << endl;
		getchar();
		return 0;
	}



	// -----------------------------6. 编译程序--------------------------------
	// 编译程序
	iStatus = clBuildProgram(Program, 1, pDevices, NULL, NULL, NULL);
	if (CL_SUCCESS != iStatus)	// 编译错误
	{
		cout << "Error: Can not build program" << endl;
		char szBuildLog[16384];
		clGetProgramBuildInfo(Program, *pDevices, CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);

		cout << "Error in Kernel: " << endl << szBuildLog;
		clReleaseProgram(Program);

		getchar();
		return 0;
	}



	//-------------------------7. 创建输入输出内存对象--------------------------------
	//每个work_item处理的数据元素个数
	size_t work_size = 100000;
	// 计算需要的work_item数量
	size_t work_item_size = (data_size + work_size - 1) / work_size;
	cout << "work_item的数量为：" << work_item_size << endl;

	// 创建kernel中第一个参数输入内存对象
	cl_mem mem_data_in = clCreateBuffer(
		Context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,	// 输入内存为只读，并可以从宿主机内存复制到设备内存
		data_size * sizeof(int),					// 输入内存空间大小
		(void *)data_in,
		&err);

	// 创建kernel中第四个参数的输出内存对象
	cl_mem mem_data_out = clCreateBuffer(
		Context,
		CL_MEM_WRITE_ONLY,					// 输出内存只能写
		work_item_size * sizeof(int),		// 输出内存空间大小
		NULL,
		NULL);

	if ((NULL == mem_data_in) || (NULL == mem_data_out))
	{
		cout << "Error creating memory objects" << err <<  endl;
		getchar();
		return 0;
	}



	//--------------------------8. 创建内核对象-------------------------------------
	cl_kernel Kernel = clCreateKernel(Program,
		"sum",  // cl文件中的入口函数
		&err);
	if (NULL == Kernel)
	{
		cout << "Error: Can not create kernel - " << err << endl;
		getchar();
		return 0;
	}



	//----------------------------9. 设置内核参数----------------------------------
	iStatus = clSetKernelArg(Kernel,
		0,		// 参数索引
		sizeof(cl_mem),
		(void *)&mem_data_in);

	iStatus |= clSetKernelArg(Kernel, 1, sizeof(cl_int), (void *)&data_size);
	
	iStatus |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void *)&work_size);

	iStatus |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void *)&mem_data_out);
	if (CL_SUCCESS != iStatus)
	{
		cout << "Error setting kernel arguments" << endl;
		getchar();
		return 0;
	}



	// --------------------------10.运行内核---------------------------------
	size_t global_work_size[1] = { work_item_size };	// 指定每个维度的work_item数量
	size_t local_work_size[1] = { work_item_size / 50 };				// 指定每个work_group包含多少个work_item
														// 利用命令队列使将再设备上执行的内核排队
	iStatus = clEnqueueNDRangeKernel(
		CommandQueue,		// 命令队列
		Kernel,				// kernel
		1,					// 数据维数，这里是一维数组
		NULL,
		global_work_size,	// 每个维度的work_item的数量，这里是一维数组，所以数组元素只有一个
		local_work_size,	// 每个work_group有250个work_item
		0,
		NULL,
		NULL);


	if (CL_SUCCESS != iStatus)
	{
		cout << "Error: Can not run kernel - " << iStatus << endl;
		getchar();
		return 0;
	}



	// ----------------------------11. 将输出读取到主机内存----------------------------------
	int *data_out = new int[work_item_size];

	iStatus = clEnqueueReadBuffer(
		CommandQueue,		// 命令队列
		mem_data_out,		// 输出内存对象
		CL_TRUE,			// 内核读取结束之前该函数不会返回
		0,
		work_item_size * sizeof(int),
		data_out,
		0,
		NULL,
		NULL);

	if (CL_SUCCESS != iStatus)
	{
		cout << "Error: Can not reading result buffer" << iStatus << endl;
		getchar();
		return 0;
	}



	// ---------------------12--输出计算结果---------------
	int area = 0;
	for (size_t i = 0; i < work_item_size; i++) {
		area += data_out[i];
	}
  cout << "AREA: " << area << endl;
  double PI = area * 8.0 / data_size;
	cout << "PI:" << PI << endl;



	// -------------------------------13. 释放资源--------------------------------
	iStatus = clReleaseKernel(Kernel);
	iStatus = clReleaseProgram(Program);
	iStatus = clReleaseMemObject(mem_data_in);
	iStatus = clReleaseMemObject(mem_data_out);
	iStatus = clReleaseCommandQueue(CommandQueue);
	iStatus = clReleaseContext(Context);

	delete[] data_out;
	delete[] data_in;
	if (NULL != pDevices)
	{
		delete[](pDevices);
		pDevices = NULL;
	}

	system("pause");//避免执行完闪退
	return 0;
}


bool convertCLFile2String(const char *pFileName, std::string& Str)
{
	size_t		uiSize = 0;
	size_t		uiFileSize = 0;
	char		*pStr = NULL;
	std::fstream fFile(pFileName, (std::fstream::in | std::fstream::binary));
	if (fFile.is_open())
	{
		fFile.seekg(0, std::fstream::end);
		uiSize = uiFileSize = (size_t)fFile.tellg();  // 获得文件大小
		fFile.seekg(0, std::fstream::beg);
		pStr = new char[uiSize + 1];
		if (NULL == pStr)
		{
			fFile.close();
			return 0;
		}
		fFile.read(pStr, uiFileSize);				// 读取uiFileSize字节
		fFile.close();
		pStr[uiSize] = '\0';
		Str = pStr;
		delete[] pStr;
		return true;
	}

	cerr << "Error: Failed to open cl file\n:" << pFileName << endl;

	return 0;
}