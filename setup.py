import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quickviewer",
    version="0.0.1",
    author="Hannah Pullen",
    author_email="hp346@cam.ac.uk",
    description="Interactive medical image viewer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
                      'ipywidgets',
                      'future_fstrings',
                      'pydicom',
                      'nibabel',
                      'numpy',
                      'matplotlib',
                      'scikit-image',
                     ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
