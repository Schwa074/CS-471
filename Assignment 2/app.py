import random

def f(x):
    # 2 - x^2
    return 2 - x**2 

def g(x):
    # (0.0051x^5) - (0.1367x^4) + (1.24x^3) - (4.456x^2) + (5.66x) - 0.287
    return (0.0051 * x**5) - (0.1367 * x**4) + (1.24 * x**3) - (4.456 * x**2) + (5.66 * x) - 0.287 


def hill_climb(minimum, maximum, step, function):
    # Initialize current state
    random_start = random.uniform(minimum, maximum) # Start from a random point within the min and max params
    x_value = random_start
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

def random_restart_hill_climb(minimum, maximum, step, function, restarts=20):
    best_x = None
    best_y = float('-inf')

    for _ in range(restarts):
        # Apply hill climbing from this random start
        x_value, y_value = hill_climb(minimum, maximum, step, function)
        
        # If this restart finds a better maximum, update the best values
        if y_value > best_y:
            best_x = x_value
            best_y = y_value

    return best_x, best_y

# Run hill-climb for f(x)
max_x, max_value = hill_climb(-5, 5, 0.5, f)
print(f"Hill climbing for f(x) found at x = {max_x:.2f}, with value f({max_x:.2f}) = {max_value:.2f} with step 0.5")

# Run hill-climb for g(x)
max_x, max_value = hill_climb(-5, 5, 0.01, f)
print(f"Hill climbing for f(x) found at x = {max_x:.2f}, with value f({max_x:.2f}) = {max_value:.2f} with step 0.01")

# Run random-restart hill-climb for g(x)
max_x, max_value = hill_climb(0, 10, 0.5, g)
print(f"Hill climb for g(x) found at x = {max_x:.2f}, with value g({max_x:.2f}) = {max_value:.2f}")

# Run random-restart hill-climb for g(x)
max_x, max_value = random_restart_hill_climb(0, 10, 0.5, g, 20)
print(f"Random restart hill climb for g(x) found at x = {max_x:.2f}, with value g({max_x:.2f}) = {max_value:.2f}")