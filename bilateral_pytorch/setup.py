from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bilateral_cuda',
    ext_modules=[
        CUDAExtension('bilateral_cuda', [
            'bilateral.cpp',
            'bilateral_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })