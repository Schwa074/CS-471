import math

def is_cluster(circles):
    '''
    Checks if a list of circles forms a cluster, where a cluster is defined as a group of circles where
    each circle overlaps with at least one other circle in the group, and all circles are reachable through overlaps.

    Args:
        circles: A list of tuples of the form (<float> x, <float> y, <float> r) where each tuple represents a circle with coordinates (x, y) and radius (r).

    Returns:
        bool: True if the given circles form a cluster and return false if they don’t form a cluster. 
    '''

    def calculate_distance(c1, c2):
        # Calculates the distance between two circle centers.
        x1, y1 = c1[:2]
        x2, y2 = c2[:2]
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance

    # Step 1: Create a graph representation of the circles.
    # Each circle is represented as a node, and an edge exists between two nodes 
    # if the corresponding circles overlap.
    # Example graph for 3 circles would be {0: [], 1: [], 2: []}
    graph = {i: [] for i in range(len(circles))} # Initialize adjacency list

    print(f"Graph init: {graph}")
    print("\n")

    # NOTE Graph is build in O(n**2) which I'm not a huge fan of but not sure how to avoid this
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):# Only consider pairs of circles once
            d = calculate_distance(circles[i], circles[j])
            r1, r2 = circles[i][2], circles[j][2]
            # Check for true overlap: edges must intersect
            if 0 < d <= r1 + r2 and d >= abs(r1 - r2):
                print(f"Graph before add: {graph}")
                graph[i].append(j) # Add edge from circle i to circle j
                graph[j].append(i) # Add edge from circle j to circle i
                print(f"Graph after add: {graph}")
                print("\n")

    # Step 2: Use depth first search to check connectivity of the graph.
    visited = set() # Set to track visited nodes (circles)

    def perform_depth_first_search(node_index):
        '''
        Recursively visits all nodes connected to the current node using depth first search.

        Args:
            node (int): The current circle index.
        '''
        print(f"Performing DFS for node index: {node_index}")
        visited.add(node_index) # Mark the current node as visited
        for neighbor in graph[node_index]: # Iterate through all connected nodes
            print(f"Looking at neighbor: {neighbor}")
            if neighbor not in visited: # If a node hasn't been visited, visit it
                print(f"Recursively calling DFS for neighbor: {neighbor} \n")
                perform_depth_first_search(neighbor)

    print(f"Final graph: {graph} \n")
    # Start depth first search from the first circle (node index 0)
    perform_depth_first_search(0)

    # Step 3: Verify if all circles are part of the same connected component.
    # If the number of visited nodes equals the total number of circles, 
    # it means all circles are connected, forming a single cluster.
    print("\n")
    print(f'Visted: {visited}')
    print(f'Circles: {circles}')
    return len(visited) == len(circles)


# Test Cases
test1 = [(1, 3, 0.7), (2, 3, 0.4), (3, 3, 0.9)] # All circles overlap directly or indirectly
print("@@@ Test 1 @@@ \n")
print(f"\n Test 1 result - {is_cluster(test1)}")  # Output: True
print("\n")

print("@@@Test 2 @@@ \n")
test2 = [(1.5, 1.5, 1.3), (4, 4, 0.7)] # Circles do not overlap
print(f"\n Test 2 result - {is_cluster(test2)}")  # Output: False
print("\n")

print("@@@ Test 3 @@@ \n")
test3 = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.1), (0.7, 0.7, 0.4), (4, 4, 0.7)] # Some circles overlap, but not all are connected
print(f"\n Test 3 result - {is_cluster(test3)}")  # Output: False
print("\n")

print("@@@ Test 4 @@@ \n")
test4 = [(4, 4, 0.5), (3, 3, 3)] # One small circle within a large circle (overlap but does not intersect edges)
print(f"\n Test 4 result - {is_cluster(test4)}")  # Output: False
print("\n")
