"""
Setup script for Hadronization Quantum Simulation Environment
Run this script to install all required packages and test the environment.
"""

import subprocess
import sys
import importlib

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

def check_package(package_name, import_name=None):
    """Check if a package is available"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is available")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT available")
        return False

def main():
    print("🔧 Setting up Quantum Hadronization Environment")
    print("=" * 50)
    
    # Core packages needed
    packages = [
        ("qiskit", "qiskit"),
        ("qiskit-aer", "qiskit_aer"),
        ("qiskit-algorithms", "qiskit_algorithms"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("jupyter", "jupyter")
    ]
    
    print("\n📋 Checking existing packages...")
    missing_packages = []
    
    for package, import_name in packages:
        if not check_package(package, import_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            install_package(package)
    else:
        print("\n🎉 All packages are already installed!")
    
    print("\n🧪 Testing quantum computing setup...")
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        import numpy as np
        
        # Create a simple test circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Test simulation
        simulator = AerSimulator()
        job = simulator.run(qc, shots=100)
        result = job.result()
        counts = result.get_counts()
        
        print("✅ Quantum circuit simulation works!")
        print(f"   Test results: {counts}")
        
    except Exception as e:
        print(f"❌ Quantum setup test failed: {e}")
        return False
    
    print("\n🚀 Environment setup complete!")
    print("\nTo use the Hadronization simulation:")
    print("1. Open: hadronization_schwinger_qiskit.ipynb")
    print("2. Select Python kernel (any available Python 3.x)")
    print("3. Run the cells sequentially")
    
    return True

if __name__ == "__main__":
    main()