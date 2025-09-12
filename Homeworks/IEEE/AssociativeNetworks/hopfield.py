# Hopfield network implementation
import numpy as np

def hopfield_network(patterns, pattern_to_test, max_iterations=1000):
    num_neurons = patterns[0].size
    # Initialize weight matrix
    weight_matrix = np.zeros((num_neurons, num_neurons), dtype=np.float32)
    # Trainging
    for pattern in patterns:
        pattern = pattern.reshape(-1,1).astype(np.float32)
        weight_matrix += np.dot(pattern, pattern.T)
    # Fill diagonal with 0's
    np.fill_diagonal(weight_matrix, 0)

    weight_matrix /= num_neurons

    current_pattern = pattern_to_test.reshape(-1, 1).astype(np.float32)
    for _ in range(max_iterations):
        prev =  current_pattern.copy()
        # Calculate the new state for all neurons simultaneously
        net_input = np.dot(weight_matrix, current_pattern)
        # Apply the sign function to all neurons at once
        current_pattern = np.where(net_input >= 0, 1, -1)
        # Check for convergence
        if np.array_equal(current_pattern, prev):
            print(f"Converged after {_ + 1} iterations.")
            return current_pattern.ravel().astype(np.int8)


    print(f"Max iterations reached. Did not converge.")
    print(current_pattern)
    return current_pattern.ravel().astype(np.int8)
