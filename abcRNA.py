import os
import numpy as np
from typing import List, Tuple, Optional
import random
from scipy.spatial.distance import pdist, squareform
from extract_rna_secondary_structure import extract_rna_sequence, extract_reference_structure
from visualize import compare_structures,kabsch_alignment
from dataclasses import dataclass
import pandas as pd

@dataclass
class OptimizationConfig:
    max_iterations: int = 2000
    colony_size: int = 100
    limit: int = 20
    alpha1: float = 0.3   # Weight for recovery
    alpha2: float = 0.2   # Weight for macro F1
    alpha3: float = 0.5   # Weight for SC-TM 
    search_space_bounds: Tuple[float, float] = (0, 10)
    initial_step_size: float = 1.0
    step_size_decay: float = 0.998  # decay for step size
    stagnation_limit: int = 100     # iterations without improvement before reset
    min_step_size: float = 0.01     # min step size
    max_sequence_length: int = 500  # Maximum allowed RNA sequence length


def kabsch_alignment(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Align structure P to Q using the Kabsch algorithm.
    Returns the rotated version of P.
    """
    # Center the structures
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    # Compute covariance matrix
    C = np.dot(P_centered.T, Q_centered)
    V, S, Wt = np.linalg.svd(C)
    d = np.linalg.det(np.dot(Wt.T, V.T))
    D = np.eye(3)
    D[2, 2] = d
    U = np.dot(np.dot(Wt.T, D), V.T)
    P_aligned = np.dot(P_centered, U)
    # Translate back to Q's centroid
    P_aligned += np.mean(Q, axis=0)
    return P_aligned


class RNATertiaryStructurePredictor:
    def __init__(self, secondary_structure: str, reference_structure: np.ndarray,
                 config: Optional[OptimizationConfig] = None):
        self.secondary_structure = secondary_structure
        self.sequence_length = len(secondary_structure)
        self.config = config or OptimizationConfig()
        self.reference_structure = reference_structure

        # Parse base pairs
        self.base_pairs = self.parse_secondary_structure()

        # Initialize optimization state
        self.food_sources = self._initialize_food_sources()
        self.trials = np.zeros(self.config.colony_size)
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.convergence_history = []

        # Adaptive parameters
        self.current_step_size = self.config.initial_step_size
        self.stagnation_counter = 0

        # Adam parameters for local refinement
        self.adam_lr = 0.01
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999

    def _initialize_food_sources(self) -> List[np.ndarray]:
        """Initialize food sources with a helix-inspired structure with perturbations."""
        sources = []
        min_val, max_val = self.config.search_space_bounds
        for _ in range(self.config.colony_size):
            structure = np.zeros((self.sequence_length, 3))
            # helical structure:
            for i in range(self.sequence_length):
                angle = 2 * np.pi * i / 11  # ~11 nucleotides per turn
                radius = 4.0
                z_step = 3.0
                structure[i] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    z_step * i / 11
                ]
            #  random perturbations
            structure += np.random.normal(0, 0.3, structure.shape)
            # Enforce consecutive nucleotide bond lengths ~4Å
            for i in range(1, self.sequence_length):
                direction = structure[i] - structure[i - 1]
                norm = np.linalg.norm(direction)
                if norm != 0:
                    direction /= norm
                structure[i] = structure[i - 1] + direction * 4.0
            sources.append(structure)
        return sources

    def parse_secondary_structure(self) -> List[Tuple[int, int]]:
        """Extract base pairs from secondary structure notation."""
        stack = []
        pairs = []
        for i, char in enumerate(self.secondary_structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    pairs.append((stack.pop(), i))
        return pairs

    def apply_constraints(self, structure: np.ndarray):
        """Apply RNA structural constraints."""
        # Maintain consecutive nucleotide distances at ~4Å
        for i in range(1, self.sequence_length):
            direction = structure[i] - structure[i - 1]
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction /= norm
            structure[i] = structure[i - 1] + direction * 4.0

        # Enforce base-pair constraints for paired bases (target ~6Å)
        for base1, base2 in self.base_pairs:
            vec = structure[base2] - structure[base1]
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            direction = vec / norm
            midpoint = (structure[base1] + structure[base2]) / 2.0
            structure[base1] = midpoint - direction * 3.0
            structure[base2] = midpoint + direction * 3.0

    def _generate_neighbor_solution(self, solution: np.ndarray) -> np.ndarray:
        new_solution = solution.copy()
        num_mutations = np.random.poisson(2)  
        for _ in range(num_mutations):
            i = random.randint(0, self.sequence_length - 1)
            j = random.randint(0, 2)
            mutation = np.random.normal(0, self.current_step_size)
            new_solution[i, j] += mutation
        # Enforce bounds
        min_val, max_val = self.config.search_space_bounds
        return np.clip(new_solution, min_val, max_val)

    def fitness_function(self, structure: np.ndarray) -> float:
        """Combined fitness with improved weighting and smoothness term."""
        recovery = self.calculate_recovery(structure)
        macro_f1 = self.calculate_macro_f1(structure)
        sc_tm = self.calculate_sc_tm(structure)
        smoothness = self._calculate_smoothness(structure)

        fitness = (self.config.alpha1 * recovery +
                   self.config.alpha2 * macro_f1 +
                   self.config.alpha3 * sc_tm +
                   0.1 * smoothness)
        return max(fitness, 0)

    def _calculate_smoothness(self, structure: np.ndarray) -> float:
        distances = pdist(structure)
        variance = np.var(distances)
        return 1.0 / (1.0 + variance)

    def calculate_recovery(self, structure: np.ndarray) -> float:
        """Recovery metric based on MSE between pairwise distance matrices."""
        distances_pred = squareform(pdist(structure))
        distances_ref = squareform(pdist(self.reference_structure))
        mse = np.mean((distances_pred - distances_ref) ** 2)
        return 1 / (1 + mse)

    def calculate_macro_f1(self, structure: np.ndarray) -> float:
        """Macro F1 calculation based on contact maps."""
        threshold = 5.0
        distances_pred = squareform(pdist(structure))
        distances_ref = squareform(pdist(self.reference_structure))
        predicted_contacts = distances_pred < threshold
        reference_contacts = distances_ref < threshold
        np.fill_diagonal(predicted_contacts, False)
        np.fill_diagonal(reference_contacts, False)
        tp = np.sum(np.logical_and(predicted_contacts, reference_contacts))
        fp = np.sum(np.logical_and(predicted_contacts, np.logical_not(reference_contacts)))
        fn = np.sum(np.logical_and(np.logical_not(predicted_contacts), reference_contacts))
        epsilon = 1e-6
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall + epsilon)

    def calculate_sc_tm(self, structure: np.ndarray) -> float:
        """SC-TM score based on RMSD (after Kabsch alignment) to reference."""
        aligned = kabsch_alignment(structure, self.reference_structure)
        rmsd = self.calculate_rmsd(aligned, self.reference_structure)
        return np.exp(-rmsd / 10)

    def calculate_rmsd(self, structure_a: np.ndarray, structure_b: np.ndarray) -> float:
        diff = structure_a - structure_b
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rmsd

    def employed_bees_phase(self) -> int:
        improvements = 0
        for i in range(self.config.colony_size):
            new_solution = self._generate_neighbor_solution(self.food_sources[i])
            if self._update_solution_if_better(i, new_solution):
                improvements += 1
        return improvements

    def onlooker_bees_phase(self) -> int:
        improvements = 0
        fitnesses = np.array([self.fitness_function(fs) for fs in self.food_sources])
        fitness_shift = fitnesses - np.min(fitnesses)
        probabilities = np.exp(fitness_shift) / np.sum(np.exp(fitness_shift))
        for _ in range(self.config.colony_size):
            i = np.random.choice(self.config.colony_size, p=probabilities)
            new_solution = self._generate_neighbor_solution(self.food_sources[i])
            if self._update_solution_if_better(i, new_solution):
                improvements += 1
        return improvements

    def scout_bees_phase(self):
        threshold = self.config.limit // 2 if self.stagnation_counter > self.config.stagnation_limit // 2 else self.config.limit
        abandoned = np.where(self.trials > threshold)[0]
        for i in abandoned:
            # Reinitialize a single source from our structured initializer
            self.food_sources[i] = self._initialize_food_sources()[i % self.config.colony_size]
            self.trials[i] = 0

    def _update_solution_if_better(self, index: int, new_solution: np.ndarray) -> bool:
        new_fitness = self.fitness_function(new_solution)
        current_fitness = self.fitness_function(self.food_sources[index])
        if new_fitness > current_fitness:
            self.food_sources[index] = new_solution
            self.trials[index] = 0
            return True
        else:
            self.trials[index] += 1
            return False

    def update_best_solution(self):
        fitnesses = [self.fitness_function(fs) for fs in self.food_sources]
        best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[best_idx]
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = self.food_sources[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Adaptive step size
        self.current_step_size *= self.config.step_size_decay
        self.current_step_size = max(self.current_step_size, self.config.min_step_size)

        self.convergence_history.append(self.best_fitness)
        if self.stagnation_counter >= self.config.stagnation_limit:
            self._partial_reset()

    def _partial_reset(self):
        print(f"Resetting at iteration {len(self.convergence_history)} due to stagnation")
        new_sources = self._initialize_food_sources()
        num_keep = max(1, self.config.colony_size // 10)
        fitnesses = np.array([self.fitness_function(fs) for fs in self.food_sources])
        top_indices = np.argsort(fitnesses)[-num_keep:]
        for i in range(self.config.colony_size):
            if i not in top_indices:
                self.food_sources[i] = new_sources[i % len(new_sources)]
        self.trials = np.zeros(self.config.colony_size)
        self.current_step_size = self.config.initial_step_size
        self.stagnation_counter = 0

    def run(self) -> Tuple[np.ndarray, float]:
        for iteration in range(self.config.max_iterations):
            emp_improvements = self.employed_bees_phase()
            onl_improvements = self.onlooker_bees_phase()
            self.scout_bees_phase()
            self.update_best_solution()

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.4f}, "
                      f"Step Size = {self.current_step_size:.4f}, ")

            # Early stopping condition if improvement is minimal over last 100 iterations
            if len(self.convergence_history) > 100:
                recent_change = self.best_fitness - self.convergence_history[-100]
                if abs(recent_change / self.best_fitness) < 1e-6 and self.best_fitness > 0:
                    print(f"Converged after {iteration} iterations")
                    break

        # After global search, apply local gradient refinement using an Adam-like optimizer
        self.best_solution = self.local_refinement(self.best_solution.copy())
        self.best_fitness = self.fitness_function(self.best_solution)
        return self.best_solution, self.best_fitness

    def local_refinement(self, structure: np.ndarray) -> np.ndarray:
        """Refine solution using an Adam-inspired gradient optimization."""
        m = np.zeros_like(structure)
        v = np.zeros_like(structure)
        epsilon = 1e-8
        t = 0
        for _ in range(100): 
            t += 1
            grad = self.compute_gradient(structure)
            m = self.adam_beta1 * m + (1 - self.adam_beta1) * grad
            v = self.adam_beta2 * v + (1 - self.adam_beta2) * (grad ** 2)
            m_hat = m / (1 - self.adam_beta1 ** t)
            v_hat = v / (1 - self.adam_beta2 ** t)
            structure += self.adam_lr * m_hat / (np.sqrt(v_hat) + epsilon)
            self.apply_constraints(structure)
        return structure

    def compute_gradient(self, structure: np.ndarray) -> np.ndarray:
        """Compute numerical gradient for the fitness function."""
        grad = np.zeros_like(structure)
        epsilon = 1e-7
        for i in range(self.sequence_length):
            for j in range(3):
                orig = structure[i, j]
                structure[i, j] = orig + epsilon
                f_plus = self.fitness_function(structure)
                structure[i, j] = orig - epsilon
                f_minus = self.fitness_function(structure)
                grad[i, j] = (f_plus - f_minus) / (2 * epsilon)
                structure[i, j] = orig
        return grad


def process_rna_files_in_folder(folder_path):
    # List all PDB files in the directory
    pdb_files = [f for f in os.listdir(folder_path) if f.endswith('.pdb')]

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['PDB_ID', 'Best_Fitness', 'Recovery_Score', 'Macro_F1_Score', 'SC_TM_Score', 'RMSD', 'Sequence_Length'])

    # Iterate over each PDB file
    for pdb_file in pdb_files:
        pdb_file_path = os.path.join(folder_path, pdb_file)
        secondary_structure = extract_rna_sequence(pdb_file_path)

        if not secondary_structure:
            print(f"Error: No RNA sequence extracted from {pdb_file}. Skipping this file.")
            continue

        # Check sequence length
        if len(secondary_structure) >= 500:
            print(f"Skipping {pdb_file}: Sequence length {len(secondary_structure)} exceeds limit of 500")
            continue

        reference_structure = extract_reference_structure(pdb_file_path)
        config = OptimizationConfig(
            max_iterations=5000,
            colony_size=100,
            initial_step_size=1.0,
            step_size_decay=0.998,
            stagnation_limit=100
        )
        predictor = RNATertiaryStructurePredictor(secondary_structure, reference_structure, config)
        best_structure, best_fitness = predictor.run()

        # Ensure alignment before RMSD
        aligned_best = kabsch_alignment(best_structure, reference_structure)
        recovery = predictor.calculate_recovery(best_structure)
        macro_f1 = predictor.calculate_macro_f1(best_structure)
        sc_tm = predictor.calculate_sc_tm(best_structure)
        rmsd = predictor.calculate_rmsd(aligned_best, reference_structure)

        # Add results to the DataFrame
        new_row = pd.DataFrame([{
            'PDB_ID': pdb_file,
            'Best_Fitness': best_fitness,
            'Recovery_Score': recovery,
            'Macro_F1_Score': macro_f1,
            'SC_TM_Score': sc_tm,
            'RMSD': rmsd,
            'Sequence_Length': len(secondary_structure)
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        print(f"Processed {pdb_file} - Length: {len(secondary_structure)} - Results added to DataFrame.")

    # Save the DataFrame to a CSV file
    results_df.to_csv('rna_structure_results.csv', index=False)
    print("All results saved to rna_structure_results.csv.")


if __name__ == "__main__":
    folder_path = "rnasolo_dataset" 
    process_rna_files_in_folder(folder_path)


""""
if __name__ == "__main__":
    pdb_file = "rnasolo_dataset/1FIR_1_A.pdb"
    secondary_structure = extract_rna_sequence(pdb_file)

    if not secondary_structure:
        print("Error: No RNA sequence extracted.")
    else:
        reference_structure = extract_reference_structure(pdb_file)
        config = OptimizationConfig(
            max_iterations=5000,
            colony_size=100,
            initial_step_size=1.0,
            step_size_decay=0.998,
            stagnation_limit=100
        )
        predictor = RNATertiaryStructurePredictor(secondary_structure, reference_structure, config)
        best_structure, best_fitness = predictor.run()

        # ensure alignment before RMSD
        aligned_best = kabsch_alignment(best_structure, reference_structure)
        recovery = predictor.calculate_recovery(best_structure)
        macro_f1 = predictor.calculate_macro_f1(best_structure)
        sc_tm = predictor.calculate_sc_tm(best_structure)
        rmsd = predictor.calculate_rmsd(aligned_best, reference_structure)

        print("\nFinal Results:")
        print("==========================")
        print("Best Predicted Structure (coordinates):")
        print(best_structure)
        print(f"\nBest Fitness: {best_fitness:.4f}")
        print(f"Recovery Score: {recovery:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"SC-TM Score: {sc_tm:.4f}")
        print(f"RMSD to Reference (after alignment): {rmsd:.4f}")

        best_structure_aligned = kabsch_alignment(best_structure, reference_structure)

        compare_structures(
            best_structure_aligned, 
            reference_structure, 
            secondary_structure, 
            "Best Predicted Structure (Aligned)", 
            "Reference Structure"
        )
"""
