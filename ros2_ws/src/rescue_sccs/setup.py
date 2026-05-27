from setuptools import setup
import os
from glob import glob

package_name = 'rescue_sccs'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'env_node    = rescue_sccs.env_node:main',
            'robot_node  = rescue_sccs.robot_node:main',
            'sccs_node   = rescue_sccs.sccs_node:main',
            'tcfm_node   = rescue_sccs.tcfm_node:main',
            'viz_node    = rescue_sccs.viz_node:main',
        ],
    },
)
