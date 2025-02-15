import math
import numpy as np
from typing import List, Tuple
import random
from scipy.spatial.distance import pdist, squareform
from extract_rna_secondary_structure import extract_rna_sequence, extract_reference_structure
from visualize import visualize_structure

class RNATertiaryStructurePredictor:
    def __init__(self, secondary_structure: str, reference_structure: np.ndarray, 
                 max_iterations: int = 5000, colony_size: int = 50, limit: int = 10, 
                 alpha1: float = 0.6, alpha2: float = 0.2, alpha3: float = 0.2, eta: float = 0.1):
        self.secondary_structure = secondary_structure
        self.sequence_length = len(secondary_structure)
        self.reference_structure = reference_structure
        self.max_iterations = max_iterations
        self.colony_size = colony_size
        self.limit = limit
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.eta = eta
        
        self.food_sources = self.initialize_food_sources()
        self.trials = np.zeros(colony_size)
        self.best_solution = None
        self.best_fitness = float('-inf')
    
    def initialize_food_sources(self) -> List[np.ndarray]:
        return [self.generate_random_structure() for _ in range(self.colony_size)]
    
    def generate_random_structure(self):
        # Generate a random 3d structure with coordinates in the range [0, 10]
        structure = np.random.rand(self.sequence_length, 3) * 10

        # bond length constraints
        for i in range(1, self.sequence_length):
            structure[i] = structure[i - 1] + np.random.normal(0, 1, 3)
        return structure

    def fitness_function(self, structure: np.ndarray) -> float:
        recovery = self.calculate_recovery(structure)
        macro_f1 = self.calculate_macro_f1(structure)
        sc_tm = self.calculate_sc_tm(structure)
        fitness = (self.alpha1 * recovery + 
                   self.alpha2 * macro_f1 + 
                   self.alpha3 * sc_tm)
        return max(fitness, 0)
    
    def calculate_recovery(self, structure: np.ndarray) -> float:
        """
        Recovery metric is computed based on the mse between the 
        pairwise distance matrices of the predicted and reference structures.
        """
        distances_pred = squareform(pdist(structure))
        distances_ref = squareform(pdist(self.reference_structure))
        mse = np.mean((distances_pred - distances_ref) ** 2)
        recovery_score = 1 / (1 + mse)
        return recovery_score
    
    def calculate_macro_f1(self, structure: np.ndarray) -> float:
        """
        calculate Macro F1 based on predicted contacts.
        A contact is defined as two nucleotides with Euclidean dist less than a threshold.
        """

        threshold = 5.0
        
        # pairwise distance matrices for both predicted and reference structures
        distances_pred = squareform(pdist(structure))
        distances_ref = squareform(pdist(self.reference_structure))
        
        # 1 if in contact, 0 otherwise
        predicted_contacts = distances_pred < threshold
        reference_contacts = distances_ref < threshold

        # eliminating self contacts by zeroing the diagonal
        np.fill_diagonal(predicted_contacts, False)
        np.fill_diagonal(reference_contacts, False)
        
        # Calculate TP fp fn
        tp = np.sum(np.logical_and(predicted_contacts, reference_contacts))
        fp = np.sum(np.logical_and(predicted_contacts, np.logical_not(reference_contacts)))
        fn = np.sum(np.logical_and(np.logical_not(predicted_contacts), reference_contacts))
        
        # Calculate precision and recall
        epsilon = 1e-6 #avoid 0
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        if (precision + recall) == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return f1
    
    def calculate_sc_tm(self, structure: np.ndarray) -> float:
        """
        SC-TM score based on RMSD between the predicted structure and the reference.
        """
        rmsd = self.calculate_rmsd(structure, self.reference_structure)
        sc_tm_score = np.exp(-rmsd / 10)
        return sc_tm_score
    
    def calculate_rmsd(self, structure_a: np.ndarray, structure_b: np.ndarray) -> float:
        # Calculate the RMSD between the two sets of coordinates
        diff = structure_a - structure_b
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rmsd
    
    def employed_bees_phase(self):
        for i in range(self.colony_size):
            new_solution = self.food_sources[i].copy()
            # Choose a random nucleotide and coordinate (dimension) to update
            j = random.randint(0, self.sequence_length - 1)
            k = random.randint(0, 2)
            # Choose another food source (partner) randomly (ensure partner not equal i)
            partner = i
            while partner == i:
                partner = random.randint(0, self.colony_size - 1)
            # Generate a perturbation using the difference between the current and partner solution
            phi = random.uniform(-self.eta, self.eta)
            new_solution[j, k] = self.food_sources[i][j, k] + phi * (self.food_sources[i][j, k] - self.food_sources[partner][j, k])
            # Ensure new coordinate is inn [0, 10]
            new_solution[j, k] = np.clip(new_solution[j, k], 0, 10)
            
            new_fitness = self.fitness_function(new_solution)
            current_fitness = self.fitness_function(self.food_sources[i])
            
            if new_fitness > current_fitness:
                self.food_sources[i] = new_solution
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def onlooker_bees_phase(self):
        fitness_values = np.array([self.fitness_function(fs) for fs in self.food_sources])
        total_fitness = np.sum(fitness_values)

        probabilities = fitness_values / total_fitness if total_fitness > 0 else np.ones(self.colony_size) / self.colony_size
        
        for _ in range(self.colony_size):
            # Probabilistically select a food source based on its fitness
            i = np.random.choice(self.colony_size, p=probabilities)
            new_solution = self.food_sources[i].copy()
            j = random.randint(0, self.sequence_length - 1)
            k = random.randint(0, 2)
            partner = i
            while partner == i:
                partner = random.randint(0, self.colony_size - 1)
            phi = random.uniform(-self.eta, self.eta) * random.uniform(0.5, 2.0)
            new_solution[j, k] = self.food_sources[i][j, k] + phi * (self.food_sources[i][j, k] - self.food_sources[partner][j, k])
            new_solution[j, k] = np.clip(new_solution[j, k], 0, 10)
            
            new_fitness = self.fitness_function(new_solution)
            current_fitness = self.fitness_function(self.food_sources[i])
            
            if new_fitness > current_fitness:
                self.food_sources[i] = new_solution
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def scout_bees_phase(self):
        # Replace food sources that haven t improved beyond the limit
        for i in range(self.colony_size):
            if self.trials[i] > self.limit:
                self.food_sources[i] = self.generate_random_structure()
                self.trials[i] = 0
    
    def update_best_solution(self):
        # Update the best solution among all food sources
        for i in range(self.colony_size):
            fitness = self.fitness_function(self.food_sources[i])
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.food_sources[i].copy()
    
    def run(self) -> Tuple[np.ndarray, float]:
        for iteration in range(self.max_iterations):
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()
            self.update_best_solution()
            
            # Print progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                current_rmsd = self.calculate_rmsd(self.best_solution, self.reference_structure)
                print(f"Iteration {iteration + 1} | Best Fitness: {self.best_fitness:.4f} | RMSD: {current_rmsd:.4f}")
        return self.best_solution, self.best_fitness

if __name__ == "__main__":
    pdb_file = "rnasolo_dataset/1FIR_1_A.pdb"
    secondary_structure = extract_rna_sequence(pdb_file)
    
    if not secondary_structure:
        print("Error: No RNA sequence extracted.")
    else:
        reference_structure = extract_reference_structure(pdb_file)
        predictor = RNATertiaryStructurePredictor(
            secondary_structure, 
            reference_structure,
            max_iterations=5000,  
            colony_size=200,       
            limit=20,              
            alpha1=0.7,        # recovery    
            alpha2=0.2,
            alpha3=0.1,
            eta=0.2                
        )
        best_structure, best_fitness = predictor.run()
        
        # Calculate detailed metrics for the best structure
        recovery = predictor.calculate_recovery(best_structure)
        macro_f1 = predictor.calculate_macro_f1(best_structure)
        sc_tm = predictor.calculate_sc_tm(best_structure)
        rmsd = predictor.calculate_rmsd(best_structure, reference_structure)
        
        print("\nFinal Results:")
        print("==========================")
        print("Best Predicted Structure (coordinates):")
        print(best_structure)
        print(f"\nBest Fitness: {best_fitness:.4f}")
        print(f"Recovery Score: {recovery:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"SC-TM Score: {sc_tm:.4f}")
        print(f"RMSD to Reference: {rmsd:.4f}")
        visualize_structure(best_structure,secondary_structure, "Best Predicted Structure")
        visualize_structure(reference_structure,secondary_structure, "Reference Structure")
