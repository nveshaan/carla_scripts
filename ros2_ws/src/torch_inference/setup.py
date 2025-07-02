from setuptools import find_packages, setup

package_name = 'torch_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=['torch_inference', 'torch_inference.models'],
    package_data={
	package_name: ['checkpoints/*.pth'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
    ],
    zip_safe=True,
    maintainer='nveshaan',
    maintainer_email='nveshaan@gmail.com',
    description='Inference Node for PyTorch Model',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'infer = torch_inference.inference_node:main',
        ],
    },
)
