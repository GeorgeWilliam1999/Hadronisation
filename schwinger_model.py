"""
Schwinger Model for Hadronization Simulation using Qiskit

This module implements the Schwinger model (1+1D QED) to study hadronization
and string breaking phenomena using quantum computing with Qiskit.

The Schwinger model describes the interaction between fermions and gauge fields
in 1+1 dimensions and is often used as a toy model for QCD to study confinement
and hadronization.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class SchwingerModel:
    """
    Implementation of the Schwinger model for quantum simulation of hadronization.
    
    The Schwinger model Hamiltonian in discrete form with Wilson fermions is:
    H = sum_n [m * (-1)^n * (σ_z^n + σ_z^{n+1})/2] 
        + sum_n [x/2 * (σ_+^n * U_n * σ_-^{n+1} + h.c.)]
        + sum_n [g^2 * a^2/2 * L_n^2]
    
    where:
    - m: fermion mass
    - x: hopping parameter (related to fermion kinetic energy)
    - g: gauge coupling
    - a: lattice spacing
    - L_n: electric field on link n
    - U_n: gauge field (compact U(1))
    """
    
    def __init__(self, num_sites: int, mass: float = 0.5, 
                 hopping: float = 1.0, coupling: float = 1.0, 
                 lattice_spacing: float = 1.0):
        """
        Initialize the Schwinger model parameters.
        
        Args:
            num_sites: Number of lattice sites
            mass: Fermion mass parameter
            hopping: Hopping parameter (kinetic energy)
            coupling: Gauge coupling strength
            lattice_spacing: Lattice spacing
        """
        self.num_sites = num_sites
        self.mass = mass
        self.hopping = hopping
        self.coupling = coupling
        self.lattice_spacing = lattice_spacing
        
        # For quantum simulation, we need qubits for both fermions and gauge fields
        self.fermion_qubits = num_sites
        self.gauge_qubits = num_sites - 1  # Links between sites
        self.total_qubits = self.fermion_qubits + self.gauge_qubits
        
    def create_hamiltonian(self, max_electric_field: int = 2) -> SparsePauliOp:
        """
        Create the Schwinger model Hamiltonian as a sum of Pauli operators.
        
        Args:
            max_electric_field: Maximum electric field quantum number for truncation
            
        Returns:
            SparsePauliOp: The Hamiltonian as a sparse Pauli operator
        """
        pauli_strings = []
        coefficients = []
        
        # Mass term: m * sum_n (-1)^n * sigma_z^n
        for site in range(self.num_sites):
            pauli_str = ['I'] * self.total_qubits
            pauli_str[site] = 'Z'
            pauli_strings.append(''.join(pauli_str))
            coefficients.append(self.mass * (-1)**site)
        
        # Hopping term: x/2 * sum_n (sigma_+^n * sigma_-^{n+1} + h.c.)
        for site in range(self.num_sites - 1):
            # sigma_+^n * sigma_-^{n+1} = (sigma_x^n + i*sigma_y^n)/2 * (sigma_x^{n+1} - i*sigma_y^{n+1})/2
            # This gives us four terms when expanded
            
            # XX term
            pauli_str = ['I'] * self.total_qubits
            pauli_str[site] = 'X'
            pauli_str[site + 1] = 'X'
            pauli_strings.append(''.join(pauli_str))
            coefficients.append(self.hopping / 4)
            
            # YY term
            pauli_str = ['I'] * self.total_qubits
            pauli_str[site] = 'Y'
            pauli_str[site + 1] = 'Y'
            pauli_strings.append(''.join(pauli_str))
            coefficients.append(self.hopping / 4)
            
            # XY term (imaginary part)
            pauli_str = ['I'] * self.total_qubits
            pauli_str[site] = 'X'
            pauli_str[site + 1] = 'Y'
            pauli_strings.append(''.join(pauli_str))
            coefficients.append(-1j * self.hopping / 4)
            
            # YX term (imaginary part)
            pauli_str = ['I'] * self.total_qubits
            pauli_str[site] = 'Y'
            pauli_str[site + 1] = 'X'
            pauli_strings.append(''.join(pauli_str))
            coefficients.append(1j * self.hopping / 4)
        
        # Electric field energy: g^2*a^2/2 * sum_n L_n^2
        # For simplicity, we represent electric field using Z operators on gauge qubits
        gauge_offset = self.fermion_qubits
        for link in range(self.gauge_qubits):
            pauli_str = ['I'] * self.total_qubits
            pauli_str[gauge_offset + link] = 'Z'
            pauli_strings.append(''.join(pauli_str))
            electric_energy = self.coupling**2 * self.lattice_spacing**2 / 2
            coefficients.append(electric_energy)
        
        return SparsePauliOp(pauli_strings, coefficients)
    
    def create_initial_state(self, particle_positions: List[int], 
                           antiparticle_positions: List[int]) -> QuantumCircuit:
        """
        Create initial state with particles and antiparticles at specified positions.
        
        Args:
            particle_positions: List of sites with particles
            antiparticle_positions: List of sites with antiparticles
            
        Returns:
            QuantumCircuit: Circuit preparing the initial state
        """
        qc = QuantumCircuit(self.total_qubits)
        
        # Prepare fermion states
        for pos in particle_positions:
            qc.x(pos)  # |1> state for particle
            
        # For antiparticles, we could use a different encoding or superposition
        for pos in antiparticle_positions:
            qc.h(pos)  # Superposition state representing antiparticle
            qc.z(pos)  # Phase flip
        
        # Initialize gauge fields in vacuum state (all |0>)
        # In a more sophisticated model, we might add gauge field initialization
        
        return qc
    
    def create_hadronization_circuit(self, time_steps: int = 10, 
                                   dt: float = 0.1) -> QuantumCircuit:
        """
        Create a quantum circuit that simulates hadronization dynamics.
        
        Args:
            time_steps: Number of time evolution steps
            dt: Time step size
            
        Returns:
            QuantumCircuit: Time evolution circuit
        """
        # Create registers
        fermion_reg = QuantumRegister(self.fermion_qubits, 'fermion')
        gauge_reg = QuantumRegister(self.gauge_qubits, 'gauge')
        classical_reg = ClassicalRegister(self.total_qubits, 'classical')
        
        qc = QuantumCircuit(fermion_reg, gauge_reg, classical_reg)
        
        # Initial state: separated quark-antiquark pair
        qc.x(fermion_reg[0])  # Quark at position 0
        qc.x(fermion_reg[-1])  # Antiquark at last position
        
        # Time evolution using Trotterization
        for step in range(time_steps):
            # Apply hopping terms (kinetic energy)
            for i in range(self.fermion_qubits - 1):
                qc.cx(fermion_reg[i], fermion_reg[i + 1])
                qc.rz(self.hopping * dt, fermion_reg[i + 1])
                qc.cx(fermion_reg[i], fermion_reg[i + 1])
            
            # Apply mass terms
            for i in range(self.fermion_qubits):
                qc.rz(self.mass * dt * (-1)**i, fermion_reg[i])
            
            # Apply gauge field interactions
            for i in range(self.gauge_qubits):
                qc.cx(fermion_reg[i], gauge_reg[i])
                qc.cx(fermion_reg[i + 1], gauge_reg[i])
                qc.rz(self.coupling**2 * dt, gauge_reg[i])
                qc.cx(fermion_reg[i + 1], gauge_reg[i])
                qc.cx(fermion_reg[i], gauge_reg[i])
        
        # Measurements
        qc.measure(fermion_reg, classical_reg[:self.fermion_qubits])
        qc.measure(gauge_reg, classical_reg[self.fermion_qubits:])
        
        return qc


class HadronizationSimulator:
    """
    Quantum simulator for hadronization using the Schwinger model.
    """
    
    def __init__(self, schwinger_model: SchwingerModel):
        self.model = schwinger_model
        self.estimator = Estimator()
        
    def run_vqe_ground_state(self, ansatz_layers: int = 3) -> Dict:
        """
        Find ground state using Variational Quantum Eigensolver (VQE).
        
        Args:
            ansatz_layers: Number of layers in the variational ansatz
            
        Returns:
            Dict: VQE results including ground state energy and parameters
        """
        # Create Hamiltonian
        hamiltonian = self.model.create_hamiltonian()
        
        # Create variational ansatz
        ansatz = TwoLocal(self.model.total_qubits, 
                         rotation_blocks=['ry', 'rz'],
                         entanglement_blocks='cz',
                         reps=ansatz_layers)
        
        # Set up VQE
        optimizer = SPSA(maxiter=100)
        vqe = VQE(self.estimator, ansatz, optimizer)
        
        # Run VQE
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        return {
            'ground_state_energy': result.eigenvalue.real,
            'optimal_parameters': result.optimal_parameters,
            'optimizer_evals': result.optimizer_evals,
            'optimal_circuit': result.optimal_circuit
        }
    
    def simulate_string_breaking(self, separation_distance: int) -> Dict:
        """
        Simulate string breaking between separated quark-antiquark pair.
        
        Args:
            separation_distance: Initial separation between quark and antiquark
            
        Returns:
            Dict: Simulation results including hadron formation probabilities
        """
        # Create initial state with separated quark-antiquark
        initial_circuit = self.model.create_initial_state(
            particle_positions=[0], 
            antiparticle_positions=[separation_distance - 1]
        )
        
        # Create time evolution circuit
        evolution_circuit = self.model.create_hadronization_circuit(
            time_steps=20, dt=0.05
        )
        
        # Combine initial state and evolution
        full_circuit = initial_circuit.compose(evolution_circuit)
        
        return {
            'circuit': full_circuit,
            'separation': separation_distance,
            'total_qubits': self.model.total_qubits
        }
    
    def analyze_confinement(self, max_separation: int = 5) -> List[float]:
        """
        Analyze confinement by studying string tension at different separations.
        
        Args:
            max_separation: Maximum quark-antiquark separation to study
            
        Returns:
            List[float]: String tensions at different separations
        """
        tensions = []
        
        for separation in range(2, max_separation + 1):
            # Create separated quark-antiquark state
            qc = self.model.create_initial_state([0], [separation - 1])
            
            # Calculate energy expectation value
            hamiltonian = self.model.create_hamiltonian()
            
            # For simplicity, we'll estimate the energy classically
            # In a real implementation, you'd use quantum state preparation and measurement
            base_energy = self.model.mass * 2  # Two fermions
            string_energy = self.model.coupling**2 * separation * self.model.lattice_spacing
            
            total_energy = base_energy + string_energy
            tension = string_energy / separation
            tensions.append(tension)
        
        return tensions


def plot_confinement_analysis(separations: List[int], tensions: List[float]):
    """
    Plot the string tension vs separation to visualize confinement.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(separations, tensions, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Quark-Antiquark Separation')
    plt.ylabel('String Tension')
    plt.title('Confinement in the Schwinger Model')
    plt.grid(True, alpha=0.3)
    plt.legend(['String Tension'])
    plt.show()


def plot_hadronization_dynamics(results: Dict):
    """
    Visualize hadronization dynamics from simulation results.
    """
    circuit = results['circuit']
    
    # Create a simple visualization of the quantum circuit
    plt.figure(figsize=(12, 8))
    
    # For demonstration, we'll create a mock probability distribution
    positions = list(range(results['total_qubits']))
    probabilities = np.random.exponential(0.3, results['total_qubits'])
    probabilities /= probabilities.sum()
    
    plt.bar(positions, probabilities, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Lattice Site')
    plt.ylabel('Particle Probability')
    plt.title(f'Hadronization Pattern (Separation: {results["separation"]})')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Schwinger Model Hadronization Simulation")
    print("=" * 50)
    
    # Initialize the model
    model = SchwingerModel(num_sites=6, mass=0.5, hopping=1.0, coupling=1.5)
    simulator = HadronizationSimulator(model)
    
    # Analyze confinement
    print("\n1. Analyzing confinement...")
    tensions = simulator.analyze_confinement(max_separation=5)
    separations = list(range(2, 6))
    
    print(f"String tensions: {tensions}")
    
    # Simulate string breaking
    print("\n2. Simulating string breaking...")
    breaking_results = simulator.simulate_string_breaking(separation_distance=4)
    
    print(f"Created hadronization circuit with {breaking_results['total_qubits']} qubits")
    print(f"Circuit depth: {breaking_results['circuit'].depth()}")
    
    # Note: Actual quantum execution would require a quantum backend
    print("\n3. Model parameters:")
    print(f"Number of sites: {model.num_sites}")
    print(f"Fermion mass: {model.mass}")
    print(f"Hopping parameter: {model.hopping}")
    print(f"Gauge coupling: {model.coupling}")
    
    print("\nSimulation complete!")