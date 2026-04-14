import os
import sys

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'maps', 'geojson', 'src', 'notebooks']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created {dir_name}/ directory")

def check_files():
    """Check if required files exist"""
    required_files = {
        'data/algeria_forest_fires.csv': 'Dataset file',
        'geojson/dz.json': 'GeoJSON file for Algeria map',
        'src/main.py': 'Main prediction script',
        'src/test.py': 'Model testing script'
    }
    
    missing = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing.append(f"  - {file_path} ({description})")
    
    if missing:
        print("\n  Missing required files:")
        print("\n".join(missing))
        print("\nPlease add these files before running the project.")
        return False
    return True

if __name__ == "__main__":
    print("Setting up Algeria Forest Fire Prediction Project...")
    create_directories()
    if check_files():
        print("\n Setup complete! You can now run:")
        print("  - python src/train_model.py  (to train the model)")
        print("  - python src/main.py         (to run the main prediction)")
        print("  - python src/test.py         (to test models)")
        print("  - python app.py              (to start the API)")