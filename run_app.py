#!/usr/bin/env python3
"""
AI College Recommender - Startup Script
This script helps you get started with the AI College Recommender application.
"""

import os
import sys
import subprocess
import time
import importlib.util

def print_banner():
    """Print the application banner"""
    print("=" * 60)
    print("🎓 AI College Recommender")
    print("=" * 60)
    print("An intelligent college recommendation system using machine learning")
    print("=" * 60)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Define package names and their import names
    package_mapping = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn', 
        'streamlit': 'streamlit',
        'scikit-learn': 'sklearn',  # scikit-learn is imported as sklearn
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'pydantic': 'pydantic',
        'xgboost': 'xgboost'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def generate_sample_data():
    """Generate sample college data"""
    print("\n📊 Generating sample college data...")
    
    try:
        # Import and run the sample data generator
        import importlib.util
        spec = importlib.util.spec_from_file_location("sample_data", "data/sample_data.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not load sample_data module")
        
        sample_data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sample_data_module)
        
        sample_data_module.save_sample_data()
        print("✅ Sample data generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating sample data: {str(e)}")
        return False

def check_data_file():
    """Check if the college data file exists"""
    data_file = "data/colleges.csv"
    if os.path.exists(data_file):
        print(f"✅ College data file found: {data_file}")
        return True
    else:
        print(f"❌ College data file not found: {data_file}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting FastAPI server...")
    print("The API will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Change to api directory and start server
        os.chdir('api')
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 'main:app', 
            '--host', '0.0.0.0', '--port', '8000', '--reload'
        ])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {str(e)}")

def start_streamlit_app():
    """Start the Streamlit application"""
    print("\n🎨 Starting Streamlit application...")
    print("The web interface will be available at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application")
    
    try:
        # Change to frontend directory and start Streamlit
        os.chdir('frontend')
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501', '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit application stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {str(e)}")

def show_instructions():
    """Show instructions for running the application"""
    print("\n📋 How to Run the Application:")
    print("=" * 40)
    print("1. Start the API server (Terminal 1):")
    print("   cd api")
    print("   uvicorn main:app --reload")
    print()
    print("2. Start the Streamlit app (Terminal 2):")
    print("   cd frontend")
    print("   streamlit run streamlit_app.py")
    print()
    print("3. Access the application:")
    print("   - Web Interface: http://localhost:8501")
    print("   - API Documentation: http://localhost:8000/docs")
    print()
    print("4. Generate sample data (if needed):")
    print("   python data/sample_data.py")

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if data exists, generate if not
    if not check_data_file():
        if not generate_sample_data():
            return
    
    print("\n🎯 Choose an option:")
    print("1. Start API server")
    print("2. Start Streamlit application")
    print("3. Show instructions")
    print("4. Generate sample data")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                start_api_server()
                break
            elif choice == '2':
                start_streamlit_app()
                break
            elif choice == '3':
                show_instructions()
                break
            elif choice == '4':
                generate_sample_data()
                break
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 