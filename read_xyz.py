import numpy as np

def read_xyz_file(path: str) -> tuple[list[str], np.ndarray]:
    """
    Read an xyz file and return atom types and coordinates.

    Args:
        path: Path to the xyz file.

    Returns:
        A tuple containing atom types (list[str]) and coordinates (np.ndarray).
    """
    atoms = []
    coordinates = []
    with open(path, "r") as file:
        for i, line in enumerate(file):
            if i < 2:
                continue
            atom, *xyz = line.split()
            atoms.append(atom)
            coordinates.append(xyz)
    
    # Ensure we always return an array with shape (n, 3)
    coords_array = np.array(coordinates, dtype=float)
    if len(coords_array) == 0:
        coords_array = coords_array.reshape(0, 3)
    
    return atoms, coords_array
