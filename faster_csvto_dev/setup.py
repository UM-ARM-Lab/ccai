from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='faster-csvto',
    install_requires=[
        'torch',
        'numpy',
    ],
    ext_modules=[CppExtension('csvgd', ['faster_csvto/cpp_retraction.cpp']),],
    cmdclass={'build_ext': BuildExtension}
)
