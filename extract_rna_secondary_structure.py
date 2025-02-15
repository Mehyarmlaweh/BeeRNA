from Bio import PDB
import numpy as np

def extract_rna_sequence(pdb_file):
    """Extracts the RNA sequence (nucleotides) from a PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", pdb_file)

    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ["A", "U", "G", "C"]:  # Only RNA bases
                    sequence += residue.get_resname()

    print(f"Extracted RNA Sequence from {pdb_file}: {sequence}")
    return sequence

def extract_reference_structure(pdb_file: str) -> np.ndarray:
    """
    Extracts the reference 3D coordinates for an RNA structure from a PDB file.
    The function uses the P atom as the representative coordinate for each nucleotide.
    If the P atom is missing, it falls back to the C4' atom.
    
    Parameters:
        pdb_file (str): Path to the PDB file.
        
    Returns:
        np.ndarray: An array of shape (N, 3) where N is the number of RNA residues.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", pdb_file)
    coordinates = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue is a standard RNA nucleotide
                if residue.get_resname() in ["A", "U", "G", "C"]:
                    if "P" in residue:
                        coordinates.append(residue["P"].get_coord())
                    elif "C4'" in residue:  # fallback if P atom is not available
                        coordinates.append(residue["C4'"].get_coord())
                    else:
                        print(f"Warning: Neither 'P' nor 'C4'' found in residue {residue.get_resname()}. Skipping.")
    
    if not coordinates:
        raise ValueError("No valid RNA nucleotide coordinates were extracted from the PDB file.")
    
    coordinates = np.array(coordinates)
    print(f"Extracted reference structure coordinates from {pdb_file}: shape {coordinates.shape}")
    return coordinates
