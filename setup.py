from setuptools import setup
from setuptools import setuptools

setup(
    name='PhonemeSER',

    version='1.0.3',

    description='Predict speech emotions from wav files.',
    long_description='Predict speech emotions from wav files. Go to Github for more details: https://github.com/tabahi/Phoneme-Converge-SER',
    author="Tabahi - Abdul Rehman",
    author_email="tabahi@hotmail.fr",
    long_description_content_type="text/markdown",
    url="https://github.com/tabahi/Phoneme-Converge-SER",
    packages=setuptools.find_packages(),
    install_requires = ['sklearn', 'pydub', 'wavio'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

    py_modules=['PhonemeSER', 'examples/predict', 'examples/example', 'Train_test', 'FormantsLib/FormantsExtract', 'FormantsLib/FormatsHDFread']
)