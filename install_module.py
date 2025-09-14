#!/usr/bin/env python3
"""
Simple script to install the slaythespire module to site-packages.
Use this if pip install -e . doesn't work properly.
"""

import os
import shutil
import glob
import site
import sys

def main():
    # Find the built .so file
    sts_dir = 'sts_lightspeed'
    patterns = [
        "slaythespire.cpython-*.so",
        "slaythespire*.so", 
        "slaythespire.so"
    ]
    
    so_file = None
    for pattern in patterns:
        so_files = glob.glob(os.path.join(sts_dir, pattern))
        if so_files:
            so_file = so_files[0]
            break
    
    if not so_file:
        print("❌ No built .so file found. Run 'cd sts_lightspeed && make' first.")
        return 1
    
    # Copy to site-packages
    site_packages = site.getsitepackages()[0]
    target = os.path.join(site_packages, 'slaythespire.so')
    
    print(f"Copying {so_file} to {target}")
    try:
        shutil.copy2(so_file, target)
        print("✅ Module installed successfully")
    except PermissionError:
        print("❌ Permission denied. Try running with sudo:")
        print(f"  sudo python3 {__file__}")
        return 1
    
    # Test import
    try:
        import slaythespire
        print("✅ Import test successful")
        return 0
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())