import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_structure(structure: np.ndarray, sequence: str, title: str):
    """
    Visualize the RNA structure in 3D, connecting consecutive nucleotides (backbone).
    
    Args:
        structure (np.ndarray): The 3D coordinates of the RNA structure.
        sequence (str): The RNA sequence.
        title (str): The title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the nucleotides as dots
    ax.scatter(structure[:, 0], structure[:, 1], structure[:, 2], color='blue', label='Nucleotides')
    
    # Connect consecutive nucleotides (backbone)
    for i in range(len(structure) - 1):
        ax.plot([structure[i, 0], structure[i + 1, 0]],
                [structure[i, 1], structure[i + 1, 1]],
                [structure[i, 2], structure[i + 1, 2]], 
                color='black', label='Backbone' if i == 0 else "")
    
    # Annotate nucleotides with their sequence
    for i, (x, y, z) in enumerate(structure):
        ax.text(x, y, z, sequence[i], color='green', fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()