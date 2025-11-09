#!/usr/bin/env python3
"""
Performance Test Script
-----------------------
Quick script to verify GPU acceleration and multithreading are working.
"""

import sys
import torch
import multiprocessing

def check_cuda():
    """Check CUDA availability and GPU info"""
    print("\n=== CUDA GPU Check ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No CUDA GPU detected. Processing will use CPU.")
        print("To install CUDA support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    return cuda_available

def check_cpu():
    """Check CPU cores for multithreading"""
    print("\n=== CPU Multithreading Check ===")
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU Cores Available: {cpu_count}")

    recommended_threads = min(4, cpu_count)
    print(f"Recommended --threads value: {recommended_threads}")

    if cpu_count >= 8:
        print("✓ High-end CPU: Excellent for parallel processing")
    elif cpu_count >= 4:
        print("✓ Mid-range CPU: Good for parallel processing")
    else:
        print("⚠️  Low CPU count: Consider --threads 2 for best results")

    return cpu_count

def check_dependencies():
    """Check required dependencies"""
    print("\n=== Dependency Check ===")
    deps = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("moviepy", "MoviePy"),
        ("librosa", "Librosa"),
        ("ultralytics", "Ultralytics YOLO"),
        ("tqdm", "tqdm")
    ]

    all_good = True
    for module, name in deps:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            all_good = False

    if not all_good:
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")

    return all_good

def performance_summary(cuda_available, cpu_count):
    """Print performance summary"""
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)

    if cuda_available:
        print("🚀 GPU Mode: ENABLED")
        print("   Expected speedup: 2-3x for tracking")
    else:
        print("💻 CPU Mode: ENABLED")
        print("   Tip: Install CUDA for 2-3x speedup")

    if cpu_count >= 4:
        print(f"⚡ Multithreading: {cpu_count} cores available")
        print(f"   Expected speedup: 2-4x for clip writing")
    else:
        print(f"⏱️  Multithreading: {cpu_count} cores (limited)")

    print("\nRecommended command:")
    threads = min(4, cpu_count)
    print(f"  python VideoHighlights.py --video match.mp4 --threads {threads}")

    if cuda_available and cpu_count >= 8:
        print("\n🎯 OPTIMAL CONFIGURATION DETECTED!")
        print("   Your system is ready for maximum performance.")
    elif cuda_available or cpu_count >= 4:
        print("\n✓ GOOD CONFIGURATION")
        print("   Your system will handle video processing well.")
    else:
        print("\n⚠️  LIMITED CONFIGURATION")
        print("   Processing will work but may be slower.")

    print("="*50 + "\n")

def main():
    print("Video Highlights Generator - Performance Test")
    print("=" * 50)

    # Run checks
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Some dependencies are missing. Install them first.")
        sys.exit(1)

    cuda_available = check_cuda()
    cpu_count = check_cpu()

    # Summary
    performance_summary(cuda_available, cpu_count)

    # Next steps
    print("Next Steps:")
    print("1. Run the GUI: python VideoHighlightsGUI.py")
    print("2. Or command line: python VideoHighlights.py --video your_video.mp4")
    print("\nFor help: python VideoHighlights.py --help")
    print()

if __name__ == "__main__":
    main()
