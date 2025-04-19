import pytest
import numpy as np
import sys
import os
import tempfile
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_energy import compute_energy

def create_temp_xyz_file(atoms, coords):
    """Create a temporary XYZ file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as tmp:
        tmp.write(f"{len(atoms)}\n")
        tmp.write("Test configuration\n")
        for atom, coord in zip(atoms, coords):
            tmp.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        temp_file_path = tmp.name
    return temp_file_path

def test_compute_energy_cu_cu():
    """Test that compute_energy correctly calculates the energy for Cu-Cu."""
    # Test setup: two Cu atoms separated by 1 unit
    atoms = ["Cu", "Cu"]
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    
    # Create temporary XYZ file
    temp_file = create_temp_xyz_file(atoms, coords)
    
    try:
        # Capture standard output to verify printed energy
        with patch('sys.stdout', new=StringIO()) as fake_out:
            compute_energy(temp_file)
            output = fake_out.getvalue().strip()
        
        # Verify that the output contains the energy
        assert output.startswith("Energy:")
        
        # Extract energy value and verify it is a number
        energy_value = float(output.split()[1])
        assert isinstance(energy_value, float)
        
        # The energy for Cu-Cu should be close to the expected value
        # (Using the same parameters as in our previous tests)
        A, XI, P, Q, R0 = 0.08550, 1.2240, 10.9600, 2.2780, 2.5562
        expected_energy = 2 * (A * np.exp(-P * (1.0 / R0 - 1)) - XI * np.exp(-Q * (1.0 / R0 - 1)))
        assert abs(energy_value - expected_energy) < 0.01
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)

def test_compute_energy_print_coords():
    """Test that compute_energy correctly prints the coordinates."""
    atoms = ["Fe", "Ni"]
    coords = [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]
    
    temp_file = create_temp_xyz_file(atoms, coords)
    
    try:
        with patch('sys.stdout', new=StringIO()) as fake_out:
            compute_energy(temp_file, print_coords=True)
            output = fake_out.getvalue().strip()
        
        # Verify that the coordinates were printed
        assert "Coordinates:" in output
        
        # Verify that all coordinates are in the output
        output_lines = output.split('\n')
        assert len(output_lines) >= 2
        assert "0.5" in output
        assert "1.5" in output
        
    finally:
        os.unlink(temp_file)

def test_compute_energy_file_not_found():
    """Test that compute_energy correctly handles nonexistent files."""
    with pytest.raises(FileNotFoundError):
        compute_energy("nonexistent_file.xyz")