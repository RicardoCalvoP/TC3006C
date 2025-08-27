import numpy as np


def create_patterns():
    zero = np.array([
                    1, 1,  1,  1,  1, 1, 1,
                    1, 1,  1,  1,  1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1, -1, -1, -1, 1, 1,
                    1, 1,  1,  1,  1, 1, 1,
                    1, 1,  1,  1,  1, 1, 1,
                    ])

    one = np.array([
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    ])

    two = np.array([
                     1, 1,  1,  1,  1,  1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                     1, 1, -1, -1, -1, -1,-1,
                     1, 1, -1, -1, -1, -1,-1,
                     1, 1, -1, -1, -1, -1,-1,
                     1, 1,  1,  1,  1,  1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                      ])

    three = np.array([
                     1, 1,  1,  1,  1,  1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                    -1, -1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                     1, 1,  1,  1,  1,  1, 1,
                      ])

    four = np.array([
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                     ])

    five = np.array([
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1, -1, -1, -1,-1,-1,
                     1, 1, -1, -1, -1,-1,-1,
                     1, 1, -1, -1, -1,-1,-1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                    -1,-1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                  ])

    six = np.array([
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1, -1, -1, -1,-1,-1,
                     1, 1, -1, -1, -1,-1,-1,
                     1, 1, -1, -1, -1,-1,-1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     ])

    seven = np.array([
                       1, 1,  1,  1,  1, 1, 1,
                       1, 1,  1,  1,  1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      -1,-1, -1, -1, -1, 1, 1,
                      ])

    eight = np.array([
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1, -1, -1, -1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     1, 1,  1,  1,  1, 1, 1,
                     ])

    nine = np.array([
                      1, 1,  1,  1,  1, 1, 1,
                      1, 1,  1,  1,  1, 1, 1,
                      1, 1, -1, -1, -1, 1, 1,
                      1, 1, -1, -1, -1, 1, 1,
                      1, 1, -1, -1, -1, 1, 1,
                      1, 1,  1,  1,  1, 1, 1,
                      1, 1,  1,  1,  1, 1, 1,
                     -1,-1, -1, -1, -1, 1, 1,
                     -1,-1, -1, -1, -1, 1, 1,
                     -1,-1, -1, -1, -1, 1, 1,
                     -1,-1, -1, -1, -1, 1, 1,
                     -1,-1, -1, -1, -1, 1, 1,
                      ])

    patterns = [zero, one, two, four, five]
    # patterns = [zero, one, two, three, four, five, six, seven, eight, nine]
    return patterns

def hopfield_network(patterns, pattern_to_test, max_iterations=1000):
    num_neurons = patterns[0].size

    # Initialize weight matrix
    weight_matrix = np.zeros((num_neurons, num_neurons))
    for pattern in patterns:
        print(pattern.reshape(12,7))
        pattern =  np.array(pattern)
        weight_matrix += np.outer(pattern, pattern)
    np.fill_diagonal(weight_matrix, 0)

    # Test with a noisy pattern
    current_pattern = pattern_to_test.copy()

    for _ in range(max_iterations):
        # Calculate the new state for all neurons simultaneously
        net_input = np.dot(weight_matrix, current_pattern)

        # Apply the sign function to all neurons at once
        new_pattern = np.where(net_input >= 0, 1, -1)

        # Check for convergence
        if np.array_equal(new_pattern, current_pattern):
            print(f"Converged after {_ + 1} iterations.")
            return new_pattern

        current_pattern = new_pattern

    print(f"Max iterations reached. Did not converge.")
    return current_pattern



if __name__ == "__main__":
    patterns = create_patterns()

    # Pass the original 'zero' pattern to the network
    original_pattern = patterns[2].copy()

    print("Starting Hopfield Network with the original 'zero' pattern.")

    # Run the network with the original pattern
    retrieved_pattern = hopfield_network(patterns, original_pattern)

    # Check the result
    print("Retrieved Pattern:")
    print(retrieved_pattern.reshape(12, 7))

    print("Original Pattern:")
    print(original_pattern.reshape(12, 7))

    if np.array_equal(retrieved_pattern, original_pattern):
        print("\nRetrieval was successful! The network converged to the original pattern.")
    else:
        print("\nRetrieval failed. The network did not converge to the original pattern.")