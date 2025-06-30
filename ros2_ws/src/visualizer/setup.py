from setuptools import find_packages, setup

package_name = 'visualizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nveshaan',
    maintainer_email='nveshaan@gmail.com',
    description='Waypoint PID visualizer in Rviz2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visualize = visualizer.visualize:main'
        ],
    },
)
