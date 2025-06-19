#!/usr/bin/env python3
"""
Setup script for AI College Recommender
Installs dependencies and prepares the environment.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "data",
        "models",
        "api",
        "frontend", 
        "utils"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/")

def generate_sample_data():
    """Generate sample college data"""
    print("📊 Generating sample college data...")
    try:
        subprocess.check_call([sys.executable, "data/sample_data.py"])
        print("✅ Sample data generated!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating data: {e}")
        return False

def main():
    """Main setup function"""
    print("🎓 AI College Recommender - Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed. Please check the error messages above.")
        return
    
    # Generate sample data
    if not generate_sample_data():
        print("❌ Failed to generate sample data.")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("1. Run: python run_app.py")
    print("2. Or manually:")
    print("   - Start API: cd api && uvicorn main:app --reload")
    print("   - Start UI: cd frontend && streamlit run streamlit_app.py")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 