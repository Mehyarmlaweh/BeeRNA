import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def compare_structures(structure1: np.ndarray, structure2: np.ndarray, 
                       sequence: str, title1: str, title2: str):
    """
    Visualize two RNA structures in the same 3D plot for comparison.
    
    Args:
        structure1 (np.ndarray): The 3D coordinates of the first RNA structure.
        structure2 (np.ndarray): The 3D coordinates of the second RNA structure.
        sequence (str): The RNA sequence (assumed to be the same for both structures).
        title1 (str): Label for the first structure.
        title2 (str): Label for the second structure.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(structure1[:, 0], structure1[:, 1], structure1[:, 2], 
               color='blue', label=title1)
    for i in range(len(structure1) - 1):
        ax.plot([structure1[i, 0], structure1[i + 1, 0]],
                [structure1[i, 1], structure1[i + 1, 1]],
                [structure1[i, 2], structure1[i + 1, 2]], 
                color='blue')
    
    ax.scatter(structure2[:, 0], structure2[:, 1], structure2[:, 2], 
               color='red', label=title2)
    for i in range(len(structure2) - 1):
        ax.plot([structure2[i, 0], structure2[i + 1, 0]],
                [structure2[i, 1], structure2[i + 1, 1]],
                [structure2[i, 2], structure2[i + 1, 2]], 
                color='red')

    # Annotate nucleotides 
    for i, (x, y, z) in enumerate(structure1):
        ax.text(x, y, z, sequence[i], color='green', fontsize=8)

    ax.set_title(f"Comparison of {title1} and {title2}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def kabsch_alignment(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Align structure P to structure Q using the Kabsch algorithm.
    Returns the rotated/translated version of P that best aligns to Q.
    """
    # 1. Center both structures at their centroids
    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    # 2. Compute covariance matrix
    C = np.dot(P_centered.T, Q_centered)

    # 3. SVD to find rotation
    V, S, Wt = np.linalg.svd(C)
    d = np.linalg.det(np.dot(Wt.T, V.T))
    D = np.eye(3)
    D[2, 2] = d  # Correct for right-handed coordinate system
    U = np.dot(np.dot(Wt.T, D), V.T)

    # 4. Rotate P_centered
    P_rotated = np.dot(P_centered, U)

    # 5. Translate P_rotated to Q's centroid
    P_aligned = P_rotated + Q_mean

    return P_aligned
