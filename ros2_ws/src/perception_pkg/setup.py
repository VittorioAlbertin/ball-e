from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('perception_pkg/models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vittorio',
    maintainer_email='vittorio@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_node = perception_pkg.yolo_node:main',
            'person_tracker = perception_pkg.person_tracker:main',
            'person_state_manager = perception_pkg.person_state_manager:main',
            'face_recognition_conditional = perception_pkg.face_recognition_conditional:main',
            'identification_coordinator = perception_pkg.identification_coordinator:main',
            'visualization_node = perception_pkg.visualization_node:main',
        ],
    },
)
