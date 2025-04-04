import sys
import subprocess
from pathlib import Path

print("Checking Tesseract OCR configuration...")

# Check Tesseract command line availability
try:
    tesseract_version = subprocess.check_output(['tesseract', '--version'], stderr=subprocess.STDOUT, text=True)
    print(f"Tesseract command line tool:\n{tesseract_version.splitlines()[0]}")
except Exception as e:
    print(f"❌ Tesseract command line tool not available: {e}")
    print("  → Make sure Tesseract OCR is installed and in your PATH")

# Check Python binding
try:
    import pytesseract
    print(f"Pytesseract version: {pytesseract.__version__}")
    print(f"Tesseract path: {pytesseract.get_tesseract_version()}")
    print("Available languages:")
    print(pytesseract.get_languages())
    print("✅ Pytesseract configuration OK")
except ImportError:
    print("❌ pytesseract module not installed. Install with: pip install pytesseract")
except Exception as e:
    print(f"❌ Error accessing pytesseract: {e}")

# Check unstructured library
try:
    from unstructured.partition.pdf import partition_pdf
    print("✅ unstructured.partition.pdf is available")
    
    # Check OCR option support
    from unstructured import __version__ as unstructured_version
    print(f"Unstructured version: {unstructured_version}")
    
    # Check if necessary packages for hi-res are installed
    import pkg_resources
    required_packages = ['pdf2image', 'pdfminer.six', 'pytesseract']
    missing = []
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages for hi-res: {', '.join(missing)}")
        print(f"  → Install with: pip install {' '.join(missing)}")
    else:
        print("✅ All required packages for hi-res are installed")
        
except ImportError:
    print("❌ unstructured module issue")

print("\nTesseract check complete.")