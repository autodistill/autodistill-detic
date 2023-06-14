import setuptools
from setuptools import find_packages
from setuptools.command.install import install
import re
import subprocess

with open("./autodistill_detic/__init__.py", 'r') as f:
    content = f.read()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

class AutodistillDetic(install):
    def run(self):
        install.run(self)
        installation_commands = """
            mkdir -p ~/.cache/autodistill/ &&
            cd ~/.cache/autodistill/ &&
            pip install 'git+https://github.com/facebookresearch/detectron2.git' &&
            git clone https://github.com/facebookresearch/Detic.git --recurse-submodules &&
            cd Detic &&
            pip install -r requirements.txt &&
            mkdir models &&
            wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
        """

        subprocess.run(installation_commands, shell=True, executable="/bin/bash")


setuptools.setup(
    name="autodistill_detic",
    version=version,
    author="Roboflow",
    author_email="support@roboflow.com",
    description="DETIC module for use with Autodistill",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill-detic",
    install_requires=[
        "torch",
        "supervision",
        "numpy",
        "autodistill",
    ],
    cmdclass={
        'install': AutodistillDetic,
    },
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
