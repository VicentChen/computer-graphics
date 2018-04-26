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

// ��cl�ļ�����תΪ�ַ���
bool convertCLFile2String(const char *pFileName, std::string& Str);

int main(int argc, char* argv[])
{
  srand((unsigned)time(NULL));

  cl_int err;
	// Ҫ��͵�int����
	size_t data_size = 100000000;
	int *data_in = new int[data_size];//1.5��Ԫ�ص�����
	for (size_t i = 0; i < data_size; i += 2) {
    data_in[i] = rand() % 250 - 125; // x
	  data_in[i + 1] = rand() % 250 - 125; // y
	}

	cl_int iStatus = 0;	// ��������״̬



	// -------------------1. ��ò�ѡ�����ƽ̨-----------------------------
	cl_uint	uiNumPlatforms = 0;		// ƽ̨����
	// ��ѯ���õ�ƽ̨������������״̬
	iStatus = clGetPlatformIDs(0, NULL, &uiNumPlatforms);
	if (CL_SUCCESS != iStatus)
	{
		cout << "Error: Getting platforms error" << endl;
		getchar();
		return 0;
	}

	cl_platform_id	Platform = NULL;	// ѡ���ƽ̨
	// ���ƽ̨��ַ
	if (uiNumPlatforms > 0)  // ����п���ƽ̨
	{
		// ����ƽ̨��Ϊƽ̨�����ڴ�ռ�
		cl_platform_id *pPlatforms = new cl_platform_id[uiNumPlatforms];

		// ��ÿ��õ�ƽ̨
		iStatus = clGetPlatformIDs(uiNumPlatforms, pPlatforms, NULL);
		Platform = pPlatforms[1];		// ��õ�һ��ƽ̨�ĵ�ַ
		delete[] pPlatforms;			// �ͷ�ƽ̨ռ�õ��ڴ�ռ�
	}

	// ���ƽ̨�汾��
	size_t	uiSize = 0;	// ƽ̨�汾�����ֽ���	
	// ���ƽ̨�汾�����ֽ���
	iStatus = clGetPlatformInfo(Platform, CL_PLATFORM_VERSION, 0, NULL, &uiSize);
	// �����ֽ���Ϊƽ̨�汾�������ڴ�ռ�
	char *pName = new char[uiSize];
	// ���ƽ̨�汾����
	iStatus = clGetPlatformInfo(Platform, CL_PLATFORM_VERSION, uiSize, pName, NULL);
	cout << pName << endl;
	delete[] pName;



	//--------------2. ��ѯGPU�豸����ѡ������豸------------------------
	cl_uint	uiNumDevices = 0;	// �豸����
								// ���GPU�豸����
	iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
	cl_device_id	*pDevices = NULL;	// �豸
	if (0 == uiNumDevices)				// ���û��GPU�豸
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;

		// ѡ��CPU��Ϊ�豸������豸��
		iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);

		// Ϊ�豸����ռ�
		pDevices = new cl_device_id[uiNumDevices];

		// ���ƽ̨
		iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_CPU, uiNumDevices, pDevices, NULL);
	}
	else
	{
		// Ϊ�豸����ռ�
		pDevices = new cl_device_id[uiNumDevices];
		iStatus = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_GPU, uiNumDevices, pDevices, NULL);
	}



	// -------------------3.�����豸������---------------------------------
	cl_context Context = NULL;	//�豸������
								// �����豸������
	Context = clCreateContext(NULL, 1, pDevices, NULL, NULL, NULL);
	if (NULL == Context)
	{
		cout << "Error: Can not create context" << endl;
		getchar();
		return 0;
	}



	// -------------------4.�����������--------------------------------------
	cl_command_queue CommandQueue = NULL;
	// ������1���豸���������
	CommandQueue = clCreateCommandQueue(Context, pDevices[0], 0, NULL);
	if (NULL == CommandQueue)
	{
		cout << "Error: Can not create CommandQueue" << endl;
		getchar();
		return 0;
	}



	// ----------------------5. �����������------------------------------
	const char	*pFileName = "pi.cl";	// cl�ļ���
	string	strSource = "";		// ���ڴ洢cl�ļ��еĴ���
								// ��cl�ļ��еĴ���תΪ�ַ���
	iStatus = convertCLFile2String(pFileName, strSource);

	const char *pSource = strSource.c_str();// ���strSourceָ��
	size_t uiArrSourceSize[] = { strlen(pSource) };	// �ַ�����С

	// �����������
	cl_program Program = clCreateProgramWithSource(Context, 1, &pSource, uiArrSourceSize, NULL);
	if (NULL == Program)
	{
		cout << "Error: Can not create program" << endl;
		getchar();
		return 0;
	}



	// -----------------------------6. �������--------------------------------
	// �������
	iStatus = clBuildProgram(Program, 1, pDevices, NULL, NULL, NULL);
	if (CL_SUCCESS != iStatus)	// �������
	{
		cout << "Error: Can not build program" << endl;
		char szBuildLog[16384];
		clGetProgramBuildInfo(Program, *pDevices, CL_PROGRAM_BUILD_LOG, sizeof(szBuildLog), szBuildLog, NULL);

		cout << "Error in Kernel: " << endl << szBuildLog;
		clReleaseProgram(Program);

		getchar();
		return 0;
	}



	//-------------------------7. ������������ڴ����--------------------------------
	//ÿ��work_item���������Ԫ�ظ���
	size_t work_size = 100000;
	// ������Ҫ��work_item����
	size_t work_item_size = (data_size + work_size - 1) / work_size;
	cout << "work_item������Ϊ��" << work_item_size << endl;

	// ����kernel�е�һ�����������ڴ����
	cl_mem mem_data_in = clCreateBuffer(
		Context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,	// �����ڴ�Ϊֻ���������Դ��������ڴ渴�Ƶ��豸�ڴ�
		data_size * sizeof(int),					// �����ڴ�ռ��С
		(void *)data_in,
		&err);

	// ����kernel�е��ĸ�����������ڴ����
	cl_mem mem_data_out = clCreateBuffer(
		Context,
		CL_MEM_WRITE_ONLY,					// ����ڴ�ֻ��д
		work_item_size * sizeof(int),		// ����ڴ�ռ��С
		NULL,
		NULL);

	if ((NULL == mem_data_in) || (NULL == mem_data_out))
	{
		cout << "Error creating memory objects" << err <<  endl;
		getchar();
		return 0;
	}



	//--------------------------8. �����ں˶���-------------------------------------
	cl_kernel Kernel = clCreateKernel(Program,
		"sum",  // cl�ļ��е���ں���
		&err);
	if (NULL == Kernel)
	{
		cout << "Error: Can not create kernel - " << err << endl;
		getchar();
		return 0;
	}



	//----------------------------9. �����ں˲���----------------------------------
	iStatus = clSetKernelArg(Kernel,
		0,		// ��������
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



	// --------------------------10.�����ں�---------------------------------
	size_t global_work_size[1] = { work_item_size };	// ָ��ÿ��ά�ȵ�work_item����
	size_t local_work_size[1] = { work_item_size / 50 };				// ָ��ÿ��work_group�������ٸ�work_item
														// �����������ʹ�����豸��ִ�е��ں��Ŷ�
	iStatus = clEnqueueNDRangeKernel(
		CommandQueue,		// �������
		Kernel,				// kernel
		1,					// ����ά����������һά����
		NULL,
		global_work_size,	// ÿ��ά�ȵ�work_item��������������һά���飬��������Ԫ��ֻ��һ��
		local_work_size,	// ÿ��work_group��250��work_item
		0,
		NULL,
		NULL);


	if (CL_SUCCESS != iStatus)
	{
		cout << "Error: Can not run kernel - " << iStatus << endl;
		getchar();
		return 0;
	}



	// ----------------------------11. �������ȡ�������ڴ�----------------------------------
	int *data_out = new int[work_item_size];

	iStatus = clEnqueueReadBuffer(
		CommandQueue,		// �������
		mem_data_out,		// ����ڴ����
		CL_TRUE,			// �ں˶�ȡ����֮ǰ�ú������᷵��
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



	// ---------------------12--���������---------------
	int area = 0;
	for (size_t i = 0; i < work_item_size; i++) {
		area += data_out[i];
	}
  cout << "AREA: " << area << endl;
  double PI = area * 8.0 / data_size;
	cout << "PI:" << PI << endl;



	// -------------------------------13. �ͷ���Դ--------------------------------
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

	system("pause");//����ִ��������
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
		uiSize = uiFileSize = (size_t)fFile.tellg();  // ����ļ���С
		fFile.seekg(0, std::fstream::beg);
		pStr = new char[uiSize + 1];
		if (NULL == pStr)
		{
			fFile.close();
			return 0;
		}
		fFile.read(pStr, uiFileSize);				// ��ȡuiFileSize�ֽ�
		fFile.close();
		pStr[uiSize] = '\0';
		Str = pStr;
		delete[] pStr;
		return true;
	}

	cerr << "Error: Failed to open cl file\n:" << pFileName << endl;

	return 0;
}