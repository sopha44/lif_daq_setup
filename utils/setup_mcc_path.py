"""
Setup script to add MCC DLL path to system PATH.
Run this before importing mcculw to ensure cbw64.dll can be found.
"""
import os
import sys

def setup_mcc_path():
    """Add MCC DAQ DLL directory to PATH."""
    # Common InstaCal installation paths
    possible_paths = [
        r"C:\Program Files (x86)\Measurement Computing\DAQ",
        r"C:\Program Files\Measurement Computing\DAQ",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if cbw64.dll exists
            dll_path = os.path.join(path, "cbw64.dll")
            if os.path.exists(dll_path):
                # Add to PATH
                if path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                print(f"Added to PATH: {path}")
                return True
    
    print("Warning: Could not find MCC DAQ installation directory")
    print("Expected locations:")
    for path in possible_paths:
        print(f"  - {path}")
    return False

if __name__ == '__main__':
    setup_mcc_path()
