from setuptools import setup, find_packages
import os
import glob

package_name = 'carla_control_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),  # Changed from [package_name] to find_packages()
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elmond',
    maintainer_email='elmond.pattanan@gmail.com',
    description='CARLA vehicle control and spawn nodes',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carla_bridge_controller = carla_control_node.carla_bridge_controller:main',
        ],
    },
)