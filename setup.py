"""Build the C++ hot-path engine extension (mceuchre_cpp).

    python setup.py build_ext --inplace
    # or, from within cpp/dev_env.ps1's environment:
    & $PY setup.py build_ext --inplace

See cpp/README.md for the MSVC/vcvars environment this needs on this machine.
"""

import sys

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# MSVC and GCC/Clang spell "use C++20" differently.
cxx_std_flag = "/std:c++20" if sys.platform == "win32" else "-std=c++20"

setup(
    name="mceuchre_cpp",
    ext_modules=[
        CppExtension(
            name="mceuchre_cpp",
            sources=[
                "cpp/bindings.cpp",
                "cpp/engine.cpp",
                "cpp/infoset.cpp",
                "cpp/solver.cpp",
                "cpp/match_equity.cpp",
                "cpp/belief.cpp",
                "cpp/subgame.cpp",
                "cpp/network.cpp",
            ],
            include_dirs=["cpp"],
            extra_compile_args=[cxx_std_flag],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
