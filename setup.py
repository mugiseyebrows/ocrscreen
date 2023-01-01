from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

base = os.path.dirname(__file__)
sourcefiles = [os.path.join(base, 'ocrscreen', n) for n in os.listdir('ocrscreen') if os.path.splitext(n)[1] in ['.pyx', '.c']]
extensions = [Extension("ocrscreen.core", sourcefiles)]
ext_modules = cythonize(extensions, language_level = "3")

setup(
    packages = find_packages(),
    name = 'ocrscreen',
    version='0.0.2',
    author="Stanislav Doronin",
    author_email="mugisbrows@gmail.com",
    url='https://github.com/mugiseyebrows/ocrscreen',
    description='ocr for recognizing text on computer screen',
    long_description = long_description,
    install_requires = ['numpy','opencv-python','Pillow'],
    entry_points = {
        'console_scripts': [
            'ocrscreen-learn = ocrscreen.learn:main',
            'ocrscreen-recognize = ocrscreen.recognize:main' 
        ]
    },
    ext_modules = ext_modules
)