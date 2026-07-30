#!/usr/bin/env python3
"""FlashKDA-Ascend: Flash Kimi Delta Attention on Ascend NPU"""

import os
import subprocess
import sys
import time

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX=" + extdir,
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DBUILD_PYBIND=True",
        ]

        # Pass through CANN arch
        arch = os.environ.get("FLASH_KDA_ASCEND_ARCH", "2201")
        cmake_args.append(f"-DCATLASS_ARCH={arch}")

        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_C", "-j"] + build_args,
            cwd=self.build_temp,
        )
        subprocess.check_call(
            ["cmake", "--install", ".", "--component", "_python_extension_lib"],
            cwd=self.build_temp,
        )


version = "0.0.1"

setup(
    name="flash_kda",
    version=version,
    description="FlashKDA: Flash Kimi Delta Attention (Ascend NPU)",
    packages=["flash_kda"],
    ext_modules=[CMakeExtension("flash_kda._C")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
