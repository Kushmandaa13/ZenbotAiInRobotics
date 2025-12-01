"""Quick check of what's installed"""

packages = {
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'torch': 'PyTorch',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'serial': 'PySerial',
}

print("\n" + "="*50)
print("CHECKING INSTALLED PACKAGES")
print("="*50 + "\n")

for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name:20} v{version}")
    except ImportError:
        print(f"❌ {name:20} NOT INSTALLED")

print("\n" + "="*50)

# Try ultralytics separately
try:
    from ultralytics import YOLO
    print("✅ Ultralytics/YOLOv8 is working!")
except ImportError:
    print("❌ Ultralytics NOT installed")
    print("\nInstall with: pip install ultralytics --no-deps")

print("="*50 + "\n")