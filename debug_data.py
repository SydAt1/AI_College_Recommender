#!/usr/bin/env python3
"""
Debug script to test data loading
"""

import os
import sys
import pandas as pd

def test_data_loading():
    """Test data loading from different paths"""
    print("Testing data loading...")
    
    # Test 1: Direct path
    direct_path = "data/colleges.csv"
    print(f"1. Testing direct path: {direct_path}")
    if os.path.exists(direct_path):
        try:
            df = pd.read_csv(direct_path)
            print(f"   ✅ Success: Loaded {len(df)} colleges")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ❌ File not found")
    
    # Test 2: Absolute path
    abs_path = os.path.abspath("data/colleges.csv")
    print(f"2. Testing absolute path: {abs_path}")
    if os.path.exists(abs_path):
        try:
            df = pd.read_csv(abs_path)
            print(f"   ✅ Success: Loaded {len(df)} colleges")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ❌ File not found")
    
    # Test 3: Data processor path
    print("3. Testing data processor path")
    try:
        sys.path.append('.')
        from models.data_processor import CollegeDataProcessor
        dp = CollegeDataProcessor()
        data = dp.load_data()
        if data is not None:
            print(f"   ✅ Success: Loaded {len(data)} colleges")
        else:
            print("   ❌ Data processor returned None")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Current working directory
    print(f"4. Current working directory: {os.getcwd()}")
    print(f"   Files in current directory: {os.listdir('.')}")
    if os.path.exists('data'):
        print(f"   Files in data directory: {os.listdir('data')}")

if __name__ == "__main__":
    test_data_loading() 