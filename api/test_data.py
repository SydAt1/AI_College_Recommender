#!/usr/bin/env python3
"""
Test data loading from API directory
"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append('..')

def test_data_loading():
    """Test data loading from API directory"""
    print("Testing data loading from API directory...")
    
    # Test 1: Direct relative path
    print("1. Testing relative path '../data/colleges.csv'")
    try:
        data = pd.read_csv('../data/colleges.csv')
        print(f"   ✅ Success: Loaded {len(data)} colleges")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Data processor
    print("2. Testing data processor")
    try:
        from models.data_processor import CollegeDataProcessor
        dp = CollegeDataProcessor()
        data = dp.load_data()
        if data is not None:
            print(f"   ✅ Success: Loaded {len(data)} colleges")
        else:
            print("   ❌ Data processor returned None")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Current directory
    print(f"3. Current directory: {os.getcwd()}")
    print(f"   Parent directory: {os.path.dirname(os.getcwd())}")
    print(f"   Data directory exists: {os.path.exists('../data')}")
    if os.path.exists('../data'):
        print(f"   Files in data directory: {os.listdir('../data')}")

if __name__ == "__main__":
    test_data_loading() 