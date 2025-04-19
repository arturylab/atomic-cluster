import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from potentials.gupta import Gupta, parameters

# Common configuration: two atoms separated by 1 unit along the x-axis
separation = 1.0
coords = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])

def calculate_expected_energy(A, XI, P, Q, R0, r):
    """Calculates the expected energy using the Gupta potential formula."""
    return 2 * (A * np.exp(-P * (r / R0 - 1)) - XI * np.exp(-Q * (r / R0 - 1)))

def calculate_expected_gradient(A, XI, P, Q, R0, r):
    """Calculates the expected gradient using the analytical derivative of the Gupta potential."""
    # Derivative of energy with respect to r
    dE_dr = 2 * (-P/R0 * A * np.exp(-P * (r/R0 - 1)) + Q/R0 * XI * np.exp(-Q * (r/R0 - 1)))
    
    # For two particles, the gradient is equal and opposite along the x direction
    return -np.array([[dE_dr, 0.0, 0.0], [-dE_dr, 0.0, 0.0]])

# Create test functions for each pair of atoms
def test_Fe_Fe():
    atoms = ["Fe", "Fe"]
    A, XI, P, Q, R0 = parameters["Fe-Fe"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)
    
def test_Fe_Co():
    atoms = ["Fe", "Co"]
    A, XI, P, Q, R0 = parameters["Fe-Co"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Fe_Ni():
    atoms = ["Fe", "Ni"]
    A, XI, P, Q, R0 = parameters["Fe-Ni"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Co_Co():
    atoms = ["Co", "Co"]
    A, XI, P, Q, R0 = parameters["Co-Co"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Co_Ni():
    atoms = ["Co", "Ni"]
    A, XI, P, Q, R0 = parameters["Co-Ni"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Ni_Ni():
    atoms = ["Ni", "Ni"]
    A, XI, P, Q, R0 = parameters["Ni-Ni"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Cu_Cu():
    atoms = ["Cu", "Cu"]
    A, XI, P, Q, R0 = parameters["Cu-Cu"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Cu_Pd():
    atoms = ["Cu", "Pd"]
    A, XI, P, Q, R0 = parameters["Cu-Pd"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Cu_Ag():
    atoms = ["Cu", "Ag"]
    A, XI, P, Q, R0 = parameters["Cu-Ag"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Cu_Pt():
    atoms = ["Cu", "Pt"]
    A, XI, P, Q, R0 = parameters["Cu-Pt"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Cu_Au():
    atoms = ["Cu", "Au"]
    A, XI, P, Q, R0 = parameters["Cu-Au"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Pd_Pd():
    atoms = ["Pd", "Pd"]
    A, XI, P, Q, R0 = parameters["Pd-Pd"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Pd_Ag():
    atoms = ["Pd", "Ag"]
    A, XI, P, Q, R0 = parameters["Pd-Ag"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Pd_Pt():
    atoms = ["Pd", "Pt"]
    A, XI, P, Q, R0 = parameters["Pd-Pt"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Pd_Au():
    atoms = ["Pd", "Au"]
    A, XI, P, Q, R0 = parameters["Pd-Au"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Ag_Ag():
    atoms = ["Ag", "Ag"]
    A, XI, P, Q, R0 = parameters["Ag-Ag"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Ag_Pt():
    atoms = ["Ag", "Pt"]
    A, XI, P, Q, R0 = parameters["Ag-Pt"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Ag_Au():
    atoms = ["Ag", "Au"]
    A, XI, P, Q, R0 = parameters["Ag-Au"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Pt_Pt():
    atoms = ["Pt", "Pt"]
    A, XI, P, Q, R0 = parameters["Pt-Pt"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Pt_Au():
    atoms = ["Pt", "Au"]
    A, XI, P, Q, R0 = parameters["Pt-Au"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_Au_Au():
    atoms = ["Au", "Au"]
    A, XI, P, Q, R0 = parameters["Au-Au"]
    gupta = Gupta(atoms)
    
    # Test energy
    energy = gupta.potential(coords)
    expected_energy = calculate_expected_energy(A, XI, P, Q, R0, separation)
    assert np.isclose(energy, expected_energy, rtol=1e-5, atol=1e-3)
    
    # Test gradient
    gradient = gupta.gradient(coords)
    expected_gradient = calculate_expected_gradient(A, XI, P, Q, R0, separation)
    assert np.allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-3)

def test_reversed_order():
    """Checks that the order of atoms does not affect the energy result."""
    # Test some pairs in both orders
    pairs = [
        (["Fe", "Co"], ["Co", "Fe"]),
        (["Cu", "Au"], ["Au", "Cu"]),
        (["Pd", "Ag"], ["Ag", "Pd"])
    ]
    
    for pair1, pair2 in pairs:
        gupta1 = Gupta(pair1)
        gupta2 = Gupta(pair2)
        
        energy1 = gupta1.potential(coords)
        energy2 = gupta2.potential(coords)
        
        assert np.isclose(energy1, energy2, rtol=1e-5, atol=1e-3)