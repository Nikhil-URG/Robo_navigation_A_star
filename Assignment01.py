import numpy as np
import math
import matplotlib.pyplot as plt

file_path = '/Users/nikhilravi/Documents/AI-Assignment01/home.txt'

home = []
with open(file_path, 'r') as file:
    for line in file:
        row = line.strip().split(' ')
        home.append(row)

initial_grid = np.array(home)

# Create a priority queue for storing the traversed paths
import heapq




# Helper function to get the heuristic
"""
We are choosing manhattan distance here because we are trying to limit the
diagonal movements and trying to move in a grid environment.
"""
def get_heuristic(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

# Helper function to find the neighbours of the given 
"""
Here we are basically looking at the neighbours and locating all of the
possible neighbours and making sure the robot is not hitting a wall or blocked path
"""
def locate_neighbours(grid, current_pos, blocked_cells = None):
    row, col = grid.shape
    # Take current node position for further calculations
    r, c = current_pos
    neighbours = []

    movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # (0, 1) - move up
    # (0, -1) - move down
    # (1, 0) - Move right
    # (-1, 0) - Move left


    for m1, m2 in movements:
        n1, n2 = r + m1, c + m2
        # Move to neighbours using the movements tuple
        # Need to check if we are within the bounds of the matrix
        if 0 <= n1 <= row and 0 <= n2 <= col and grid[n1, n2] != 'W':
            neighbour_coordinate = (n1, n2)
            if blocked_cells is None or neighbour_coordinate not in blocked_cells:
                neighbours.append(neighbour_coordinate)
    return neighbours

"""
Here we implement the A* algorithm
"""
def implement_a_star(grid, start, goal, blocked_cells = None):
    # row, col = grid.shape

    # Define the priority queue for storing the nodes that need to be explored
    open_set = [(0, start)]
    # Use a dictionary to store the path so that once goal is found we can reconstruct the path
    prev_path = {}
    # Maintain the g and f scores
    g_score = {start : 0}
    f_score = {start : get_heuristic(start, goal)}

    while open_set:
        # Extract the f_score and coordinates of the current node
        current_f_score, current_node = heapq.heappop(open_set)

        if current_node == goal: # Case when goal node is found in the current neighbours
            # Reconstruct the path
            path = []
            while current_node in prev_path:
                path.append(current_node)
                current_node = prev_path[current_node]
            path.append(start)
            return path[::-1]
        # Here we are sending the current node to get their neighbours.
        
        for neighbour in locate_neighbours(grid, current_node, blocked_cells):
            tentative_g = g_score[current_node] + 1 # We are assuming cost 1 for moving to each neighbour

            # Case when we are seeing a new node and the node has lower cost
            if neighbour not in g_score or tentative_g < g_score[neighbour]:
                prev_path[neighbour] = current_node
                g_score[neighbour] = tentative_g
                f_score[neighbour] = g_score[neighbour] + get_heuristic(neighbour, goal)
                heapq.heappush(open_set, (f_score[neighbour], neighbour))
    # Just return none if no path is found
    return None



        


    

def locate_persons_goals_robot(grid):
    persons = {}
    goals = {}
    robot = []
    count = 0
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell.isdigit():
                persons[int(cell)] = (i, j)
            if cell == 'H':
                goals[str(cell) + str(count)] = (i, j)
                count += 1
            if cell == 'r':
                robot.append((i, j))
    return goals, persons, robot




# def a_astar(grid, start, goal):

def update_robot_position(grid, current_coordinates, cell_coordinate):
    for coordinates in current_coordinates:
        x, y = coordinates
        # Update each of the current robot coordinates with . making it a free space
        grid[x][y] = '.'
    x, y = cell_coordinate
    grid[x, y] = 'r'
    grid[x - 1, y] = 'r'
    grid[x - 1, y - 1] = 'r'
    grid[x, y - 1] = 'r'
    new_robot_pos = [(x, y), (x - 1, y), (x - 1, y - 1), (x, y - 1)]

    return grid, new_robot_pos


def update_grid(grid, replace_coordinate, replace_char):
    x, y = replace_coordinate
    grid[x, y] = replace_char
    return grid


def visualize_char_grid(char_array):
    """
    Displays a 2D NumPy array (or list of lists) of characters in a Matplotlib window
    using text annotations with a monospaced font.

    Args:
        char_array (list of lists or np.ndarray): The 2D array of characters.
    """
    rows, cols = np.array(char_array).shape

    fig, ax = plt.subplots()

    # Remove axes and ticks
    ax.axis('off')

    # Use a monospaced font for alignment
    font = {'family': 'monospace',
            'size': 10}  # Adjust size as needed

    # Add text annotations for each character
    for i in range(rows):
        for j in range(cols):
            ax.text(j, -i, char_array[i][j], ha='center', va='center', fontdict=font)

    # Adjust plot limits to fit the grid
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-rows + 0.5, 0.5)

    plt.title("Character Array Display")
    plt.tight_layout()
    plt.show()



# print(home)
# get locations of chairs, persons and the robot
goals, persons, robot = locate_persons_goals_robot(initial_grid)
# Find the shortest path from robot to each person
min_path_length = float('inf')
first_person = ...
path_to_person_dict = {}
for person, person_loc in persons.items():
    path_to_person = implement_a_star(initial_grid, robot[0], person_loc)
    path_to_person_dict[person] = path_to_person

    # print(f"Path from Robot to Person {person}: {path_to_person}")
    if len(path_to_person) < min_path_length:
        min_path_length = len(path_to_person)
        first_person = person

print(f"Robot co-ordinates : {robot}")
print(f"First chosen person : {first_person}")
print(f"First person co-ordinates : {persons[first_person]}")










# print("Grid initially : ")
# # print(initial_grid)
# visualize_char_grid(initial_grid)

# # Update the robot position to the first person
# updated_grid_after_first_person, robot_position_updated = update_robot_position(initial_grid, robot, persons[first_person])

# print("Grid after the update : ")
# # updated_grid_after_first_person = update_grid(initial_grid, robot[], 'r')

# visualize_char_grid(updated_grid_after_first_person)

# # For the first person now take him to nearest chair
# min_path_length = float('inf')
# chosen_goal = ...
# person_to_goal_dict = {}
# for goal, goal_loc in goals.items():
#     path_to_goals = implement_a_star(updated_grid_after_first_person, robot_position_updated[0], goal_loc)

#     if len(path_to_goals) < min_path_length:
#         min_path_length = len(path_to_person)
#         chosen_goal = goal

# grid_after_person_goal = update_grid(updated_grid_after_first_person, goals[chosen_goal], first_person)


# print(f"First chosen goal : {chosen_goal}")
# print(f"First goal co-ordinates : {goals[chosen_goal]}")

# visualize_char_grid(grid_after_person_goal)

# # remove dictionary entry of the goal you just reached
# del goals[chosen_goal]
# del person[first_person]


# --------------------------------- For loop implementation ----------------------------------------------

print("Grid initially : ")
visualize_char_grid(initial_grid)

robot_position = robot  # Initialize robot position

person_to_goal_dict = {}  # Keep track of which person is assigned to which goal

while persons and goals:
    closest_person = None
    min_distance_to_robot = float('inf')

    for person, person_loc in persons.items():
        # Use A* to find the distance from the robot to the person
        path_to_person = implement_a_star(initial_grid.copy(), robot_position[0], person_loc)
        distance = len(path_to_person) if path_to_person else float('inf') # Important: Handle no path!

        if distance < min_distance_to_robot:
            min_distance_to_robot = distance
            closest_person = person

    if closest_person:
        print(f"\nProcessing closest person: {closest_person} at {persons[closest_person]}")

        # Update the robot position to the closest person
        updated_grid, robot_position = update_robot_position(initial_grid.copy(), robot_position, persons[closest_person])
        print("Grid after robot moves to person:")
        visualize_char_grid(updated_grid)

        min_path_length = float('inf')
        chosen_goal = None

        for goal, goal_loc in goals.items():
            path_to_goal = implement_a_star(updated_grid.copy(), robot_position[0], goal_loc)

            if path_to_goal:  # Ensure a path was found
                path_length = len(path_to_goal)  # The length of the path from the person's location to the goal
                if path_length < min_path_length:
                    min_path_length = path_length
                    chosen_goal = goal

        if chosen_goal:
            print(f"Chosen goal for {closest_person}: {chosen_goal} at {goals[chosen_goal]}")
            grid_after_assignment = update_grid(updated_grid.copy(), goals[chosen_goal], closest_person)
            print("Grid after person reaches goal:")
            visualize_char_grid(grid_after_assignment)

            person_to_goal_dict[closest_person] = chosen_goal

            # Remove the assigned person and goal
            del persons[closest_person]
            del goals[chosen_goal]

            # The updated grid becomes the initial grid for the next iteration
            initial_grid = grid_after_assignment
        else:
            print(f"No path found from robot to any available goal for {closest_person}.")
            break  # Or handle this case differently
    else:
        print("No persons left to process.")
        break

print("\nAll possible goals have been assigned (or no more paths found).")
print("Person to Goal Assignments:", person_to_goal_dict)