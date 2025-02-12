def f(x):
    return 2 - x**2

def g(x):
    return (0.0051 * x**5) - (0.1367 * x**4) + (1.24 * x**3) - (4.456 * x**2) + (5.66 * x) - 0.287


def hill_climb(minimum, maximum, step, function):
    # Initialize current state
    x_value = minimum
    y_value = function(x_value)

    # Loop to search for the maximum value
    while True:
        # List of neighbor_x_values
        neighbor_x_values = [x_value - step, x_value + step]
        
        # Filter neighbor_x_values to stay within bounds
        neighbor_x_values = [x for x in neighbor_x_values if minimum <= x <= maximum]
        
        # Evaluate the neighbor_x_values
        neighbor_y_values = [function(x) for x in neighbor_x_values]
        
        # If no better neighbor exists, we stop
        if not neighbor_y_values or max(neighbor_y_values) <= y_value:
            break
        
        # Move to the neighbor with the maximum value
        best_neighbor = neighbor_y_values.index(max(neighbor_y_values))
        x_value = neighbor_x_values[best_neighbor]
        y_value = neighbor_y_values[best_neighbor]
    
    return x_value, y_value

# Run hill-climb for f(x)
max_x, max_value = hill_climb(-5, 5, 0.5, f)
print(f"Maximum for f(x) found at x = {max_x:.2f}, with value f(x) = {max_value:.2f}")

# Run hill-climb for g(x)
max_x2, max_value2 = hill_climb(0, 10, 0.5, g)
print(f"Maximum for g(x) found at x = {max_x2:.2f}, with value g(x) = {max_value2:.2f}")