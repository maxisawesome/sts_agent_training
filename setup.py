#!/usr/bin/env python3
"""
Setup script for STS Neural Agent package.

This enables proper Python packaging and eliminates the need for
sys.path.insert statements throughout the codebase.
"""

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import shutil
import os
import glob

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return requirements
    return []

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "STS Neural Agent - Reinforcement Learning for Slay the Spire"

def build_sts_lightspeed():
    """Build sts_lightspeed using CMake."""
    print("Building sts_lightspeed with CMake...")
    
    sts_dir = os.path.join(os.path.dirname(__file__), 'sts_lightspeed')
    
    if not os.path.exists(sts_dir):
        raise RuntimeError(f"sts_lightspeed directory not found: {sts_dir}")
    
    print(f"Building in directory: {sts_dir}")
    print(f"Python executable: {os.sys.executable}")
    
    # Clean previous builds
    subprocess.run(['make', 'clean'], cwd=sts_dir, check=False)
    
    # Configure with CMake
    cmake_cmd = ['cmake', '-DPYTHON_EXECUTABLE=' + os.sys.executable, '.']
    print(f"Running: {' '.join(cmake_cmd)}")
    result = subprocess.run(cmake_cmd, cwd=sts_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CMake configure failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise RuntimeError("CMake configuration failed")
    
    # Build with make
    make_cmd = ['make', '-j4']
    print(f"Running: {' '.join(make_cmd)}")
    result = subprocess.run(make_cmd, cwd=sts_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Make failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise RuntimeError("Make build failed")
    
    print("✅ sts_lightspeed built successfully")

def copy_module_to_site_packages():
    """Copy the built module to site-packages."""
    import site
    
    sts_dir = os.path.join(os.path.dirname(__file__), 'sts_lightspeed')
    
    # Try multiple patterns to find the built .so file
    patterns = [
        "slaythespire.cpython-*.so",  # Generic pattern
        "slaythespire*.so",           # Fallback pattern
        "slaythespire.so"             # Simple pattern
    ]
    
    so_file = None
    for pattern in patterns:
        so_files = glob.glob(os.path.join(sts_dir, pattern))
        if so_files:
            so_file = so_files[0]
            break
    
    if not so_file:
        # Debug: list all files in sts_lightspeed directory
        all_files = os.listdir(sts_dir)
        so_files_debug = [f for f in all_files if f.endswith('.so')]
        raise RuntimeError(f"Could not find built module. Available .so files: {so_files_debug}")
    
    site_packages = site.getsitepackages()[0]
    target = os.path.join(site_packages, 'slaythespire.so')
    
    print(f"Copying {so_file} to {target}")
    shutil.copy2(so_file, target)
    print("✅ Module installed successfully")

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        build_sts_lightspeed()
        copy_module_to_site_packages()

class PostInstallCommand(install):
    """Post-installation for installation mode.""" 
    def run(self):
        install.run(self)
        build_sts_lightspeed()
        copy_module_to_site_packages()

setup(
    name="sts-neural-agent",
    version="0.1.0",
    author="STS Neural Agent Team", 
    description="Reinforcement Learning for Slay the Spire using neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/sts-neural-agent",
    packages=find_packages(),
    py_modules=[
        "sts_neural_network",
        "sts_training", 
        "sts_data_collection",
        "sts_reward_functions",
        "sts_neural_agent",
        "sts_model_manager",
        "train_sts_agent",
        "setup_wandb",
        "test_nn_interface",
        "test_wandb",
        "analyze_rewards",
        "simple_reward_analysis",
    ],
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "sts-train=train_sts_agent:main",
            "sts-setup-wandb=setup_wandb:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
    keywords="reinforcement-learning machine-learning slay-the-spire game-ai neural-networks ppo",
)