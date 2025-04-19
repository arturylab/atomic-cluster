import pytest
import numpy as np
import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from read_xyz import read_xyz_file

def test_read_xyz_file():
    """Test that the read_xyz_file function correctly reads XYZ files."""
    
    # Create a temporary XYZ test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as tmp:
        tmp.write("3\n")  # Number of atoms
        tmp.write("Test molecules\n")  # Comment line
        tmp.write("Cu 0.0 0.0 0.0\n")  # Atom 1
        tmp.write("Fe 1.0 0.0 0.0\n")  # Atom 2
        tmp.write("Ni 0.0 1.0 0.0\n")  # Atom 3
        temp_file_path = tmp.name
    
    try:
        # Read the file using the function to be tested
        atoms, coords = read_xyz_file(temp_file_path)
        
        # Verify atom types
        expected_atoms = ["Cu", "Fe", "Ni"]
        assert atoms == expected_atoms
        
        # Verify coordinates
        expected_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        assert np.allclose(coords, expected_coords)
        
        # Verify shapes match
        assert len(atoms) == coords.shape[0]
        assert coords.shape == (3, 3)
        
    finally:
        # Delete the temporary file
        os.unlink(temp_file_path)

def test_read_xyz_file_error_handling():
    """Test that read_xyz_file correctly handles non-existent files."""
    
    # Attempt to read a non-existent file
    with pytest.raises(FileNotFoundError):
        read_xyz_file("non_existent_file.xyz")

def test_read_xyz_file_empty():
    """Test that read_xyz_file handles files with no atoms."""
    
    # Create an empty XYZ file (header only)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as tmp:
        tmp.write("0\n")  # Zero atoms
        tmp.write("Empty file\n")  # Comment line
        temp_file_path = tmp.name
    
    try:
        atoms, coords = read_xyz_file(temp_file_path)
        
        # Verify that the lists are empty
        assert len(atoms) == 0
        assert coords.shape == (0, 3)  # Shape should be (0, 3)
        
    finally:
        os.unlink(temp_file_path)