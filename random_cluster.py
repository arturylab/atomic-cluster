# Generate random coordinates for a given number of atoms and create an xyz file.
import sys
import re
import numpy as np
from write_xyz import write_xyz_file
from potentials.gupta import parameters

def get_valid_elements():
    """
    Extract all valid elements from the Gupta parameters.
    
    Returns:
        Set of valid element symbols.
    """
    valid_elements = set()
    for key in parameters.keys():
        elem1, elem2 = key.split('-')
        valid_elements.add(elem1)
        valid_elements.add(elem2)
    return valid_elements

# Get the valid elements once at module level
VALID_ELEMENTS = get_valid_elements()

def parse_atom_sequence(sequence: str) -> list[str]:
    """
    Parse an atom sequence like Fe2Co10Ni or FeCoNi into a list of atoms.
    Only accepts valid elements defined in the Gupta potential parameters.
    
    Args:
        sequence: input sequence (e.g., Fe2Co10Ni).
    Returns:
        List of atoms (e.g., ['Fe', 'Fe', 'Co', 'Co', ..., 'Ni']).
    
    Raises:
        ValueError: If an invalid element is detected.
    """
    # Regex to match atom symbols and optional counts
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    atoms = []
    
    for match in pattern.finditer(sequence):
        atom = match.group(1)  # Atom symbol (e.g., Fe, Co, Ni)
        
        # Verify the atom is valid
        if atom not in VALID_ELEMENTS:
            valid_elements_str = ", ".join(sorted(VALID_ELEMENTS))
            raise ValueError(f"Invalid element '{atom}'. Valid elements are: {valid_elements_str}")
        
        count = match.group(2)  # Optional count (e.g., 2, 10)
        count = int(count) if count else 1  # Default to 1 if no count is provided
        atoms.extend([atom] * count)  # Add the atom 'count' times to the list
    
    # Verify that there is at least one atom
    if not atoms:
        raise ValueError("No valid atoms specified in the sequence")
    
    return atoms

def verify_atom_combinations(atoms: list[str]) -> bool:
    """
    Verify that all atom pairs in the list have defined parameters.
    
    Args:
        atoms: List of atom symbols.
    
    Returns:
        True if all combinations are valid, raises ValueError otherwise.
    """
    unique_atoms = set(atoms)
    
    # Check all possible pairs
    for i, atom1 in enumerate(sorted(unique_atoms)):
        for atom2 in sorted(unique_atoms)[i:]:
            key1 = f"{atom1}-{atom2}"
            key2 = f"{atom2}-{atom1}"
            
            if key1 not in parameters and key2 not in parameters:
                raise ValueError(f"No parameters defined for interaction between {atom1} and {atom2}")
    
    return True

def generate_random_coordinates(num_atoms: int, min_distance: float = 1.0, box_size: float = 10.0, max_attempts: int = 1000) -> np.ndarray:
    """
    Generate random coordinates for a given number of atoms ensuring a minimum distance between them.
    
    Args:
        num_atoms: Number of atoms.
        min_distance: Minimum distance allowed between any two atoms (in Å).
        box_size: Size of the cubic box in which atoms are placed (in Å).
        max_attempts: Maximum number of attempts to place an atom.
    
    Returns:
        Array of random coordinates with minimum distance constraint.
        
    Raises:
        ValueError: If unable to place all atoms with the minimum distance constraint.
    """
    # Initialize coordinates array
    coords = np.zeros((num_atoms, 3))
    
    # For each atom
    for i in range(num_atoms):
        attempt = 0
        placed = False
        
        # Increase box size if there are many atoms
        effective_box_size = box_size * (1.0 + num_atoms / 50.0)
        
        while not placed and attempt < max_attempts:
            # Generate random coordinate
            candidate = np.random.uniform(-effective_box_size/2, effective_box_size/2, 3)
            
            # If it's the first atom, place it directly
            if i == 0:
                coords[0] = candidate
                placed = True
                break
            
            # Verify distance with already placed atoms
            too_close = False
            for j in range(i):
                distance = np.linalg.norm(candidate - coords[j])
                if distance < min_distance:
                    too_close = True
                    break
            
            # If it satisfies the distance constraint, place the atom
            if not too_close:
                coords[i] = candidate
                placed = True
            
            attempt += 1
        
        # If unable to place the atom after max_attempts
        if not placed:
            raise ValueError(f"Unable to place atom {i+1} with minimum distance of {min_distance} Å after {max_attempts} attempts. "
                            "Try reducing the number of atoms or the minimum distance.")
    
    return coords

def main():
    if len(sys.argv) != 2:
        print("Usage: python random_cluster.py <atom_sequence>")
        print("Example: python random_cluster.py Fe2Co10Ni")
        
        # Display valid elements
        valid_elements_str = ", ".join(sorted(VALID_ELEMENTS))
        print(f"\nValid elements: {valid_elements_str}")
        sys.exit(1)
    
    try:
        # Use get to avoid IndexError
        atom_sequence = sys.argv[1] if len(sys.argv) > 1 else ""
        atoms = parse_atom_sequence(atom_sequence)
        
        # Verify that all atom combinations are valid
        verify_atom_combinations(atoms)
        
        num_atoms = len(atoms)
        
        # Generate random coordinates with minimum distance
        coords = generate_random_coordinates(num_atoms, min_distance=1.0)
        
        # Create the output file name
        output_file = f"rnd-{atom_sequence.lower()}.xyz"
        
        # Create the xyz file
        write_xyz_file(output_file, atoms, coords, comment="Randomly generated coordinates with min distance 1.0 Å")
        
        print(f"XYZ file '{output_file}' created successfully with {num_atoms} atoms.")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
