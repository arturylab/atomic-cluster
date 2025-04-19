import pytest
import numpy as np
import sys
import os
import tempfile
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_forces import compute_gradient
from potentials.gupta import Gupta

def create_temp_xyz_file(atoms, coords):
    """Create a temporary XYZ file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as tmp:
        tmp.write(f"{len(atoms)}\n")
        tmp.write("Test configuration\n")
        for atom, coord in zip(atoms, coords):
            tmp.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        temp_file_path = tmp.name
    return temp_file_path

def test_compute_gradient_full_output():
    """Test that compute_gradient correctly prints the full gradient matrix."""
    # Test setup: two Cu atoms separated by 1 unit
    atoms = ["Cu", "Cu"]
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    
    # Create temporary XYZ file
    temp_file = create_temp_xyz_file(atoms, coords)
    
    try:
        # Capture standard output to verify
        with patch('sys.stdout', new=StringIO()) as fake_out:
            compute_gradient(temp_file, norm=False)
            output = fake_out.getvalue().strip()
        
        # Verify that the output contains the gradient matrix
        assert "Gradient Matrix:" in output
        
        # Verify that the output is a matrix (contains multiple lines)
        assert output.count('\n') >= 1
        
        # Compute the gradient directly for comparison
        gupta = Gupta(atoms)
        expected_gradient = gupta.gradient(np.array(coords))
        
        # Verify that each component of the gradient is present in the output
        # This is sufficient to confirm that the full gradient is displayed
        for row in expected_gradient:
            for value in row:
                # Convert each value to a string with limited precision
                value_str = f"{value:.4f}"
                # Verify only that the first digits are in the output
                # (to avoid issues with exact formatting)
                assert value_str.rstrip('0').rstrip('.') in output.replace('-', ' -')
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)

def test_compute_gradient_norm():
    """Test that compute_gradient correctly calculates the gradient norm."""
    atoms = ["Cu", "Cu"]
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    
    temp_file = create_temp_xyz_file(atoms, coords)
    
    try:
        with patch('sys.stdout', new=StringIO()) as fake_out:
            compute_gradient(temp_file, norm=True)
            output = fake_out.getvalue().strip()
        
        # Verify that the output contains "Gradient Norm:"
        assert "Gradient Norm:" in output
        
        # Extract the numeric value
        norm_value = float(output.split(":")[1].strip())
        
        # Compute the norm directly for comparison
        gupta = Gupta(atoms)
        gradient = gupta.gradient(np.array(coords))
        expected_norm = np.linalg.norm(gradient)
        
        # Verify that the values match
        assert np.isclose(norm_value, expected_norm, rtol=1e-5)
        
    finally:
        os.unlink(temp_file)

def test_compute_gradient_file_not_found():
    """Test that compute_gradient correctly handles non-existent files."""
    with pytest.raises(FileNotFoundError):
        compute_gradient("non_existent_file.xyz", norm=False)

def test_compute_gradient_different_atoms():
    """Test that compute_gradient works with different types of atoms."""
    atoms = ["Fe", "Ni"]
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    
    temp_file = create_temp_xyz_file(atoms, coords)
    
    try:
        # Verify that no exceptions are raised
        with patch('sys.stdout', new=StringIO()):
            compute_gradient(temp_file, norm=True)
            # No need to verify the output, just ensure it runs without errors
        
    finally:
        os.unlink(temp_file)