#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void errCheck(cl_int err, int line);

#define CHECKERR(err) errCheck(err, __LINE__)

int main() {
    using cplx_t = std::complex<float>;
    const int sample_size = 4096;

    const int radix = 32;

    const int is_inverse = true;

    const size_t global_item_size = 128; // There can be at most sample_size / radix size work items. Must be a power of 2
    const size_t local_item_size = global_item_size;

    const char* program_source_path = "fft.cl";

    const char* kernel_entry_point = "fft";

    std::vector<cplx_t> inData(sample_size);

    for (int i = 0; i < inData.size(); i++) inData[i] = i / 10.0;

    std::ifstream fileSource(program_source_path);

    if (!fileSource) {
        std::cout << "Failed to load kernel\n";
        return 1;
    }

    std::string sourceStr((std::istreambuf_iterator<char>(fileSource)), std::istreambuf_iterator<char>());

    fileSource.close();

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_int ret;

    CHECKERR(clGetPlatformIDs(1, &platform_id, nullptr));

    CHECKERR(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr));

    cl_ulong size;
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);

    auto s2 = sample_size * sizeof(cplx_t);

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

    CHECKERR(ret);

    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, nullptr, &ret);

    CHECKERR(ret);

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sample_size * sizeof(cplx_t), nullptr, &ret);

    CHECKERR(ret);

    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, sample_size * sizeof(cplx_t), inData.data(), 0, nullptr, nullptr);

    CHECKERR(ret);

    const char* source_str = sourceStr.c_str();
    const std::size_t source_size = sourceStr.size();

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

    CHECKERR(ret);

    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

    if (ret != CL_SUCCESS) {
        std::size_t log_size;

        CHECKERR(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));

        std::string build_log(log_size, '\0');

        CHECKERR(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, const_cast<char*>(build_log.c_str()), nullptr));

        std::cout << "Building the program failed, build log:\n" << build_log << "\n";

        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, kernel_entry_point, &ret);

    CHECKERR(ret);

    CHECKERR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj)); // data

    CHECKERR(clSetKernelArg(kernel, 1, sample_size * sizeof(cplx_t), nullptr)); // local_cache

    CHECKERR(clSetKernelArg(kernel, 2, sizeof(sample_size), &sample_size)); // N

    CHECKERR(clSetKernelArg(kernel, 3, sizeof(radix), &radix)); // radix

    CHECKERR(clSetKernelArg(kernel, 4, sizeof(is_inverse), &is_inverse)); // is_inverse

    CHECKERR(clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_item_size, &local_item_size, 0, nullptr, nullptr));

    auto outData = inData;
    CHECKERR(clEnqueueReadBuffer(command_queue, a_mem_obj, CL_TRUE, 0, sample_size * sizeof(cplx_t), outData.data(), 0, nullptr, nullptr));

    for (int i = 0; i < outData.size(); i++) {
        std::cout << outData[i] << "\n";
    }

    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(a_mem_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}

const char* getErrorString(cl_int error);

void errCheck(cl_int err, int line) {
    if (err != CL_SUCCESS) {
        std::cout << "Error on line " << line << ", error: " << getErrorString(err) << "\n";
        exit(1);
    }
}

const char* getErrorString(cl_int error)
{
    switch (error) {
        // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}
