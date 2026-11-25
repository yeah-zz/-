"""Compile the official CPU DiffVG backend with local Visual Studio."""
from pathlib import Path
import shutil
import subprocess
import os

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path(r"E:\Codex_Workspace_Data\CLIPascene_20261021")
LOCAL=STORAGE/"build_tools"/"cmake"
for key,value in {"TEMP":STORAGE/"temp","TMP":STORAGE/"temp","PIP_CACHE_DIR":STORAGE/"cache"/"pip"}.items():
    os.environ[key]=str(value)
cmake=str(LOCAL/"bin"/"cmake.exe")
src=STORAGE/"source"/"diffvg-master"
cmake_file=src/"CMakeLists.txt"
cmake_text=cmake_file.read_text(encoding="utf-8")
cmake_text=cmake_text.replace("COMPONENTS Development REQUIRED", "COMPONENTS Interpreter Development REQUIRED")
cmake_text=cmake_text.replace("add_compile_options(/Wall /Zi)", "add_compile_options(/W3 /Zi)")
cmake_text=cmake_text.replace("set(SRCS atomic.h", "set(SRCS diagnostic.cpp atomic.h")
cmake_file.write_text(cmake_text,encoding="utf-8")
header=src/"diffvg.h"
header.write_text(header.read_text(encoding="utf-8").replace("inline double log2(double x)","inline double diffvg_legacy_log2(double x)"),encoding="utf-8")
parallel=src/"parallel.cpp"
parallel.write_text(parallel.read_text(encoding="utf-8").replace("int ret = std::thread::hardware_concurrency();","int ret = 4;").replace("int ret = 1;","int ret = 4;"),encoding="utf-8")
cpp=src/"diffvg.cpp"
cpp_text=cpp.read_text(encoding="utf-8")
if "clipascene_native_diagnostic" not in cpp_text:
    diagnostic=r'''
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#pragma comment(lib, "dbghelp.lib")
static LONG CALLBACK clipascene_native_diagnostic(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    HANDLE process=GetCurrentProcess();
    SymInitialize(process, nullptr, TRUE);
    void* stack[48];
    unsigned count=CaptureStackBackTrace(0,48,stack,nullptr);
    for(unsigned i=0;i<count;i++) {
        char storage[sizeof(SYMBOL_INFO)+512] = {};
        SYMBOL_INFO* symbol=(SYMBOL_INFO*)storage;
        symbol->SizeOfStruct=sizeof(SYMBOL_INFO);symbol->MaxNameLen=511;
        DWORD64 displacement=0;
        if(SymFromAddr(process,(DWORD64)stack[i],&displacement,symbol)) {
            IMAGEHLP_LINE64 line={};line.SizeOfStruct=sizeof(line);DWORD offset=0;
            if(SymGetLineFromAddr64(process,(DWORD64)stack[i],&offset,&line))
                fprintf(stderr,"NATIVE %s %s:%lu\n",symbol->Name,line.FileName,line.LineNumber);
            else fprintf(stderr,"NATIVE %s\n",symbol->Name);
        }
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
'''
    (src/"diagnostic.cpp").write_text(diagnostic+'\nvoid install_clipascene_native_diagnostic() { AddVectoredExceptionHandler(1, clipascene_native_diagnostic); }\n',encoding="utf-8")
    cpp_text=cpp_text.replace("namespace py = pybind11;","void install_clipascene_native_diagnostic();\nnamespace py = pybind11;")
    cpp_text=cpp_text.replace("PYBIND11_MODULE(diffvg, m) {","PYBIND11_MODULE(diffvg, m) {\n    install_clipascene_native_diagnostic();")
    cpp.write_text(cpp_text,encoding="utf-8")
elif "#ifndef NOMINMAX" in cpp_text:
    start=cpp_text.index("#ifndef NOMINMAX")
    end=cpp_text.index("namespace py = pybind11;",start)
    diagnostic=cpp_text[start:end]
    (src/"diagnostic.cpp").write_text(diagnostic+'\nvoid install_clipascene_native_diagnostic() { AddVectoredExceptionHandler(1, clipascene_native_diagnostic); }\n',encoding="utf-8")
    cpp_text=cpp_text[:start]+"void install_clipascene_native_diagnostic();\n"+cpp_text[end:]
    cpp_text=cpp_text.replace("AddVectoredExceptionHandler(1, clipascene_native_diagnostic);","install_clipascene_native_diagnostic();")
    cpp.write_text(cpp_text,encoding="utf-8")
build=LOCAL.parent/"diffvg-ninja-build-v3"
deps=STORAGE/"environment"/"deps"
vc=Path(r"E:\编程\Common7\Tools\VsDevCmd.bat")
raw=subprocess.check_output(f'cmd /d /u /c call "{vc}" -arch=x64 >nul && set')
env=os.environ.copy()
env.update({key.upper():value for key,value in (line.split("=",1) for line in raw.decode("utf-16-le").splitlines() if "=" in line and not line.startswith("="))})
compiler=Path(r"E:\编程\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe")
ninja=deps/"ninja"/"data"/"bin"/"ninja.exe"
sdk_bins=sorted(Path(r"C:\Program Files (x86)\Windows Kits\10\bin").glob("*/x64/rc.exe"))
sdk_bin=sdk_bins[-1].parent
subprocess.run([cmake,"-S",str(src),"-B",str(build),"-G","Ninja",
    "-DCMAKE_MAKE_PROGRAM="+ninja.as_posix(),"-DCMAKE_C_COMPILER="+compiler.as_posix(),"-DCMAKE_CXX_COMPILER="+compiler.as_posix(),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc /D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR",
    "-DCMAKE_RC_COMPILER="+(sdk_bin/"rc.exe").as_posix(),"-DCMAKE_MT="+(sdk_bin/"mt.exe").as_posix(),
    "-DDIFFVG_CUDA=OFF","-DPython_ROOT_DIR=E:/Anaconda","-DPYTHON_EXECUTABLE=E:/Anaconda/python.exe",
    "-DPYTHON_INCLUDE_PATH=E:/Anaconda/include","-DPYTHON_LIBRARY=E:/Anaconda/libs/python39.lib",
    "-DTHRUST_INCLUDE_DIR="+(src/"thrust").as_posix(),"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE="+deps.as_posix(),
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE="+deps.as_posix()],check=True,env=env)
subprocess.run([cmake,"--build",str(build),"--config","Release","--parallel","8"],check=True,env=env)
