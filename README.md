# Quantum Hadronization with the Schwinger Model

This project demonstrates **hadronization simulation** using the **Schwinger model** (1+1D QED) implemented with **Qiskit** quantum computing. The Schwinger model serves as a simplified toy model for Quantum Chromodynamics (QCD), allowing us to study fundamental phenomena like confinement and hadronization on quantum computers.

## Overview

The Schwinger model describes the interaction between fermions (quarks) and gauge fields (gluons) in 1+1 dimensions. This project implements:

- **Quantum Hamiltonian Construction**: Building the Schwinger model using Pauli operators
- **Time Evolution**: Using Trotterization to simulate hadronization dynamics
- **VQE Ground State Calculation**: Finding ground states with variational quantum eigensolvers
- **String Breaking Simulation**: Modeling confinement and hadron formation
- **Observable Measurements**: Tracking particle numbers, correlations, and energy

## Physical Motivation

In QCD, when quarks are separated, the energy stored in the color flux tube (string) eventually becomes large enough to create new quark-antiquark pairs, leading to hadronization. The Schwinger model captures this essential physics in a setting suitable for quantum simulation.

### Key Features

- ✅ **Confinement**: Linear potential between quarks
- ✅ **String Breaking**: Creation of new particle pairs
- ✅ **Hadronization**: Formation of bound states (hadrons)
- ✅ **Quantum Dynamics**: Full quantum mechanical evolution

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Hadronisation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Create a virtual environment**:
   ```bash
   python -m venv hadronization_env
   # Windows:
   hadronization_env\\Scripts\\activate
   # Linux/Mac:
   source hadronization_env/bin/activate
   pip install -r requirements.txt
   ```

## Quick Start

### Running the Jupyter Notebook

The easiest way to explore hadronization simulation:

```bash
jupyter notebook hadronization_schwinger_qiskit.ipynb
```

This notebook contains a complete tutorial with:
- Theoretical background
- Step-by-step implementation
- Quantum circuit construction
- Results visualization

### Using the Python Module

For custom simulations, use the `schwinger_model.py` module:

```python
from schwinger_model import SchwingerModel, HadronizationSimulator

# Initialize the model
model = SchwingerModel(
    num_sites=6,
    mass=0.5,
    hopping=1.0,
    coupling=1.5
)

# Create simulator
simulator = HadronizationSimulator(model)

# Run VQE for ground state
results = simulator.run_vqe_ground_state()
print(f"Ground state energy: {results['ground_state_energy']}")

# Simulate string breaking
breaking_results = simulator.simulate_string_breaking(separation_distance=4)
```

## Example Results

The simulation produces several key observables:

### String Tension Analysis
- **Linear confinement**: String tension grows with quark separation
- **Critical separation**: String breaking threshold
- **Hadron formation**: Bound state creation

### Quantum Circuit Metrics
- **Circuit depth**: ~100-500 gates depending on parameters
- **Qubit requirements**: 6-12 qubits for realistic simulations
- **Gate fidelity**: Compatible with NISQ devices

### Physical Observables
- **Particle number density**: `⟨n_i⟩ = ⟨(1 + σ_z^i)/2⟩`
- **Correlation functions**: `⟨σ_i^+ σ_j^-⟩`
- **Energy expectation**: `⟨H⟩`

## The Schwinger Model

### Hamiltonian

The discrete Schwinger model Hamiltonian with Wilson fermions:

```
H = Σ_n [m/2 * (-1)^n * (σ_z^n + σ_z^{n+1})] +
    Σ_n [x/2 * (σ_+^n * U_n * σ_-^{n+1} + h.c.)] +
    Σ_n [g²a²/2 * L_n²]
```

Where:
- `m`: fermion mass
- `x`: hopping parameter (kinetic energy)
- `g`: gauge coupling strength
- `a`: lattice spacing
- `L_n`: electric field on link n
- `U_n`: gauge field (compact U(1))

### Quantum Implementation

1. **Fermion qubits**: Represent matter fields
2. **Gauge qubits**: Represent gauge fields
3. **Trotterization**: Decompose time evolution
4. **Measurement**: Extract physical observables

## 🔬 Advanced Usage

### Custom Parameter Studies

```python
# Study confinement vs coupling strength
couplings = [0.5, 1.0, 1.5, 2.0]
results = {}

for g in couplings:
    model = SchwingerModel(num_sites=6, coupling=g)
    tensions = model.analyze_confinement(max_separation=5)
    results[g] = tensions
```

### VQE Optimization

```python
# Use different ansätze and optimizers
from qiskit_algorithms.optimizers import COBYLA, SPSA

simulator = HadronizationSimulator(model)
vqe_results = simulator.run_vqe_ground_state(
    ansatz_layers=4,
    optimizer=COBYLA(maxiter=200)
)
```

### Time Evolution Studies

```python
# Custom time evolution
circuit = model.create_hadronization_circuit(
    time_steps=50,
    dt=0.02
)
```

## Performance

### Classical Simulation Limits
- **6 qubits**: Fast simulation (seconds)
- **10 qubits**: Moderate simulation (minutes)
- **>12 qubits**: Requires HPC or quantum hardware

### Quantum Hardware
- **NISQ compatible**: Circuit depths suitable for current quantum computers
- **Error mitigation**: Can benefit from error correction
- **Scalability**: Linear scaling with system size

## Applications

This simulation framework can be extended for:

- **Lattice gauge theories**: Other gauge groups (SU(2), SU(3))
- **Higher dimensions**: 2+1D and 3+1D field theories
- **Thermal states**: Finite temperature simulations
- **Phase transitions**: Critical phenomena studies
- **Quantum algorithms**: Improved simulation techniques

## References

### Scientific Background
1. Schwinger, J. (1962). "Gauge Invariance and Mass. II." Physical Review 128, 2425.
2. Kogut, J. & Susskind, L. (1975). "Hamiltonian formulation of Wilson's lattice gauge theories."
3. Martinez, E. A. et al. (2016). "Real-time dynamics of lattice gauge theories with a few-qubit quantum computer."

### Quantum Computing
1. Nielsen & Chuang. "Quantum Computation and Quantum Information"
2. Qiskit Documentation: https://qiskit.org/
3. Quantum Algorithm Zoo: https://quantumalgorithmzoo.org/

## Contributing

Contributions are welcome! Areas for improvement:

- **New observables**: Additional physical measurements
- **Optimization**: Better quantum circuits and algorithms
- **Extensions**: Higher dimensions or different models
- **Visualization**: Enhanced plotting and analysis tools

### Development Setup

```bash
git clone <repository-url>
cd Hadronisation
pip install -r requirements.txt
pip install -e .  # Development installation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **IBM Qiskit Team**: For the excellent quantum computing framework
- **Lattice QCD Community**: For theoretical foundations
- **Quantum Computing Researchers**: For algorithmic innovations

---
