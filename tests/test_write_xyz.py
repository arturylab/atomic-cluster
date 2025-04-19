import pytest
import numpy as np
import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from write_xyz import write_xyz_file
from read_xyz import read_xyz_file

def test_write_xyz_file():
    """Test que la función write_xyz_file escribe correctamente archivos XYZ."""
    
    # Datos de prueba
    atoms = ["Cu", "Fe", "Ni"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    comment = "Estructura de prueba"
    
    # Crear un nombre temporal para el archivo
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
        temp_file_path = tmp.name
    
    try:
        # Escribir los datos en el archivo
        write_xyz_file(temp_file_path, atoms, coords, comment)
        
        # Verificar que el archivo existe
        assert os.path.exists(temp_file_path)
        
        # Leer el contenido del archivo y verificarlo línea por línea
        with open(temp_file_path, 'r') as f:
            lines = f.readlines()
            
            # Verificar número de átomos
            assert lines[0].strip() == "3"
            
            # Verificar línea de comentario
            assert lines[1].strip() == comment
            
            # Verificar datos de átomos
            assert "Cu" in lines[2] and "0.0   0.0   0.0" in lines[2]
            assert "Fe" in lines[3] and "1.0   0.0   0.0" in lines[3]
            assert "Ni" in lines[4] and "0.0   1.0   0.0" in lines[4]
        
        # Usar read_xyz_file para verificar la consistencia
        read_atoms, read_coords = read_xyz_file(temp_file_path)
        assert read_atoms == atoms
        assert np.allclose(read_coords, coords)
        
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def test_write_xyz_file_empty():
    """Test que write_xyz_file maneja correctamente el caso de cero átomos."""
    
    atoms = []
    coords = np.array([], dtype=float).reshape(0, 3)
    comment = "Archivo vacío"
    
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
        temp_file_path = tmp.name
    
    try:
        # Escribir archivo vacío
        write_xyz_file(temp_file_path, atoms, coords, comment)
        
        # Verificar contenido
        with open(temp_file_path, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip() == "0"
            assert lines[1].strip() == comment
            assert len(lines) == 2  # Solo dos líneas en total
        
        # Verificar consistencia con read_xyz_file
        read_atoms, read_coords = read_xyz_file(temp_file_path)
        assert len(read_atoms) == 0
        assert read_coords.size == 0
        
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def test_write_xyz_file_single_atom():
    """Test que write_xyz_file maneja correctamente el caso de un solo átomo."""
    
    atoms = ["Au"]
    coords = np.array([[2.5, 3.5, 4.5]])
    
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
        temp_file_path = tmp.name
    
    try:
        # Escribir archivo con un solo átomo
        write_xyz_file(temp_file_path, atoms, coords)
        
        # Verificar contenido
        with open(temp_file_path, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip() == "1"
            assert "Au" in lines[2] and "2.5   3.5   4.5" in lines[2]
        
        # Verificar consistencia
        read_atoms, read_coords = read_xyz_file(temp_file_path)
        assert read_atoms == atoms
        assert np.allclose(read_coords, coords)
        
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)