from setuptools import find_packages, setup

package_name = 'interaction_pkg'

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
            'people_database_node = interaction_pkg.people_database_node:main',
            'face_enrollment_node = interaction_pkg.face_enrollment_node:main',
            'enroll_face = interaction_pkg.enroll_face_cli:main',
        ],
    },
)
