#!/usr/bin/env python3
"""Test runner script that properly sets up the Python path."""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_test(test_path):
    """Run a specific test file."""
    print(f"\n{'='*60}")
    print(f"Running: {test_path}")
    print(f"{'='*60}")
    
    try:
        # Change to project root for consistent imports
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Run the test
        result = subprocess.run([sys.executable, test_path], 
                              capture_output=False, 
                              cwd=project_root)
        
        os.chdir(original_cwd)
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test: {e}")
        return False

def main():
    """Run all tests."""
    test_files = [
        "tests/unit/modules/test_cf_module.py",
        "tests/unit/modules/test_cca_module.py", 
        "tests/integration/modules/test_pipeline_integration.py",
        "tests/integration/modules/test_modules.py",
        "tests/unit/replayer/test_intervention_directly.py",
        "tests/functional/comprehensive_test.py"
    ]
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_path = project_root / test_file
        if test_path.exists():
            if run_test(test_file):
                passed += 1
                print(f"✅ {test_file} PASSED")
            else:
                failed += 1
                print(f"❌ {test_file} FAILED")
        else:
            print(f"⚠️  {test_file} not found")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)