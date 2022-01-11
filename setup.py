import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="jduimage",
    version="0.1.1",
    scripts=["bin/jduimage"],
    author="Jean Demeusy",
    author_email="dev.jdu@gmail.com",
    description="Simple opencv interface to simplify image processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeandemeusy/jdu-image",
    packages=["jduimage"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["opencv-python", "numpy"],
)
