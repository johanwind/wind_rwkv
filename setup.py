from setuptools import find_packages, setup

setup(
    name='wind_rwkv',
    version='0.2',
    description='Optimized kernels for RWKV models',
    author='Johan Sokrates Wind',
    author_email='johanswi@math.uio.no',
    url='https://github.com/johanwind/wind_rwkv',
    packages=find_packages(),
    package_data={
        "wind_rwkv.rwkv7.backstepping_longhead": ["*.cu", "*.cpp"],
        "wind_rwkv.rwkv7.backstepping_smallhead": ["*.cu", "*.cpp"],
        "wind_rwkv.rwkv7.chunked_cuda": ["*.cu", "*.cpp", "*.cuh"],
    },
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
    install_requires=[
        'triton>=3.0',
        'ninja'
    ]
)
