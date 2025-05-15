from setuptools import setup, find_packages

setup(
    name="auglib",
    version="0.1.0",
    description="Image augmentation pipeline",
    author="Sofya Borisova",
    author_email="saborisova_1@miem.hse.ru",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.0',
        'opencv-python>=4.11.0.86',
        'pandas>=2.2.2',
        'Augmentor>=0.2.12'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    package_dir={'auglib': 'auglib'},
    include_package_data=True,
)