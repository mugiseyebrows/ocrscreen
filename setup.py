from setuptools import setup, find_packages
from Cython.Build import cythonize

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    packages = find_packages(),
    name = 'ocrscreen',
    version='0.0.2',
    author="Stanislav Doronin",
    author_email="mugisbrows@gmail.com",
    url='https://github.com/mugiseyebrows/ocrscreen',
    description='',
    long_description = long_description,
    install_requires = ['numpy','opencv-python','Pillow'],
    entry_points={
        'console_scripts': [
            'ocrscreen-learn = ocrscreen.learn:main',
            'ocrscreen-recognize = ocrscreen.recognize:main' 
        ]
    },
    ext_modules = cythonize("ocrscreen/core.pyx", language_level = "3")
)