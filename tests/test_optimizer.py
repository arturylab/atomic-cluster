import pytest
import numpy as np
import os
import tempfile
import sys

# Add the parent directory to the system path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer import optimize_structure
from read_xyz import read_xyz_file
from write_xyz import write_xyz_file
from potentials.gupta import Gupta

def create_temp_xyz_file(atoms, coords):
    """Create a temporary XYZ file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as tmp:
        tmp.write(f"{len(atoms)}\n")
        tmp.write("Test configuration\n")
        for atom, coord in zip(atoms, coords):
            tmp.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        return tmp.name

def test_optimize_structure_local():
    """Test to verify local optimization methods."""
    # Create a simple structure that is not in equilibrium
    atoms = ["Cu", "Cu"]
    # Non-optimal separation for Cu-Cu (greater than equilibrium distance)
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    
    # Test different local optimization methods
    local_methods = ["L-BFGS-B", "BFGS", "CG", "Nelder-Mead"]
    
    for method in local_methods:
        temp_file = create_temp_xyz_file(atoms, coords)
        output_file = f"opt-{os.path.basename(temp_file)}"
        
        try:
            # Run optimization with higher tolerance for testing
            optimize_structure(temp_file, method=method, tol=1e-4, maxiter=100)
            
            # Verify that the output file exists
            assert os.path.exists(output_file)
            
            # Read the optimized structure
            opt_atoms, opt_coords = read_xyz_file(output_file)
            
            # Verify that the structure has changed
            assert not np.allclose(coords, opt_coords)
            
            # Create a Gupta instance to evaluate energies
            gupta = Gupta(atoms)
            
            # Calculate energies before and after optimization
            initial_energy = gupta.potential(coords)
            final_energy = gupta.potential(opt_coords)
            
            # Verify that the energy has decreased
            assert final_energy < initial_energy
            
            # Measure the distance between atoms
            original_distance = np.linalg.norm(coords[1] - coords[0])
            optimized_distance = np.linalg.norm(opt_coords[1] - opt_coords[0])
            
            # The distance should approach the equilibrium value (~2.5Å for Cu-Cu)
            assert optimized_distance < original_distance
            
        finally:
            # Clean up temporary files
            for file in [temp_file, output_file]:
                if os.path.exists(file):
                    os.unlink(file)

def test_optimize_structure_global():
    """Test to verify global optimization methods."""
    # More complex structure to test global optimization
    atoms = ["Cu", "Cu", "Cu"]
    # Irregular triangle (non-optimal)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 3.0, 0.0]
    ])
    
    # Test only one global method with reduced parameters for testing
    global_methods = ["basinhopping"]  # shgo and differential_evolution are too slow for routine tests
    
    for method in global_methods:
        temp_file = create_temp_xyz_file(atoms, coords)
        output_file = f"opt-{os.path.basename(temp_file)}"
        
        try:
            # Use faster configurations for testing
            optimize_structure(temp_file, method=method, tol=1e-3, maxiter=10)
            
            # Verify that the output file exists
            assert os.path.exists(output_file)
            
            # Read the optimized structure
            opt_atoms, opt_coords = read_xyz_file(output_file)
            
            # Verify that the structure has changed
            assert not np.allclose(coords, opt_coords)
            
            # Create a Gupta instance to evaluate energies
            gupta = Gupta(atoms)
            
            # Calculate energies before and after optimization
            initial_energy = gupta.potential(coords)
            final_energy = gupta.potential(opt_coords)
            
            # Verify that the energy has decreased
            assert final_energy < initial_energy
            
        finally:
            # Clean up temporary files
            for file in [temp_file, output_file]:
                if os.path.exists(file):
                    os.unlink(file)

def test_optimize_structure_error_handling():
    """Test to verify error handling in optimize_structure."""
    # Verify handling of non-existent file
    with pytest.raises(FileNotFoundError):
        optimize_structure("non_existent_file.xyz")
    
    # For invalid method test, use two atoms instead of one
    temp_file = create_temp_xyz_file(["Cu", "Cu"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    
    try:
        # Verify that the correct exception is raised for an invalid method
        with pytest.raises(ValueError):
            optimize_structure(temp_file, method="invalid_method")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)