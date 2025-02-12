def f(x):
    return 2 - x**2


def hill_climb(minimum, maximum, step):
    # Initialize current state
    x_value = minimum
    y_value = f(x_value)

    # Loop to search for the maximum value
    while True:
        # List of neighbors
        neighbors = [x_value - step, x_value + step]
        
        # Filter neighbors to stay within bounds
        neighbors = [x for x in neighbors if minimum <= x <= maximum]
        
        # Evaluate the neighbors
        neighbor_y_values = [f(x) for x in neighbors]
        
        
        # If no better neighbor exists, we stop
        if not neighbor_y_values or max(neighbor_y_values) <= y_value:
            break
        
        # Move to the neighbor with the maximum value
        best_neighbor = neighbor_y_values.index(max(neighbor_y_values))
        x_value = neighbors[best_neighbor]
        y_value = neighbor_y_values[best_neighbor]
    
    return x_value, y_value

max_x, max_value = hill_climb(-5, 5, 0.5)

print(f"Maximum found at x = {max_x:.2f}, with value f(x) = {max_value:.2f}")