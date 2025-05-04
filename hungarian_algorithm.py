import numpy as np

def solve_hungarian_algorithm(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the assignment problem using the Hungarian algorithm.
    
    Args:
        cost_matrix (np.ndarray): The cost matrix for the assignment problem.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The row and column indices of the optimal assignment.
    """
    
    cost_matrix_copy = cost_matrix.copy()
    
    # Check if the cost matrix is NxN
    is_square = cost_matrix.shape[0] == cost_matrix.shape[1]
    if not is_square:
        max_size = max(cost_matrix.shape)
        square_matrix = np.full((max_size, max_size), 1e6)
        square_matrix[:cost_matrix.shape[0], :cost_matrix.shape[1]] = cost_matrix
        cost_matrix = square_matrix
    
    print("Cost matrix:\n", cost_matrix)
    
    # Row normalization
    
    row_min = cost_matrix.min(axis=1)
    cost_matrix = cost_matrix - row_min[:, np.newaxis]
    
    # Column normalization
    col_min = cost_matrix.min(axis=0)
    cost_matrix = cost_matrix - col_min
    
    print("Normalized cost matrix:\n", cost_matrix)
    
    n = cost_matrix.shape[0]
    while True:
        # Greedy assignment (1 zero per row, different column)
        assignment_matrix = greedy_assignment(cost_matrix)

        if assignment_matrix.sum() == n:
            break

        # Step 4: Cover zeros with minimum number of lines
        covered_rows, covered_cols = find_min_line_cover(cost_matrix, assignment_matrix)

        # Step 5: Modify matrix
        min_uncovered = np.inf
        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    min_uncovered = min(min_uncovered, cost_matrix[i][j])

        for i in range(n):
            for j in range(n):
                if i not in covered_rows and j not in covered_cols:
                    cost_matrix[i][j] -= min_uncovered
                elif i in covered_rows and j in covered_cols:
                    cost_matrix[i][j] += min_uncovered
                # other cells stay the same

    # Final assignment extraction
    row_indices, col_indices = [], []
    for i in range(cost_matrix_copy.shape[0]):
        for j in range(cost_matrix_copy.shape[1]):
            if assignment_matrix[i][j] == 1:
                row_indices.append(i)
                col_indices.append(j)

    return np.array(row_indices), np.array(col_indices)

def greedy_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Greedily assign tasks to workers based on the cost matrix.
    This improved version tries to maximize the number of assignments.
    
    Args:
        cost_matrix (np.ndarray): The cost matrix for the assignment problem.
        
    Returns:
        np.ndarray: The assignment matrix with 1s at assigned positions.
    """
    n, m = cost_matrix.shape
    assignment_matrix = np.zeros((n, m))
    
    # Find all positions with zeros
    zero_positions = []
    for i in range(n):
        for j in range(m):
            if cost_matrix[i, j] == 0:
                # Store (row, col, count of zeros in row & col)
                row_zeros = np.sum(cost_matrix[i, :] == 0)
                col_zeros = np.sum(cost_matrix[:, j] == 0)
                zero_positions.append((i, j, row_zeros + col_zeros))
    
    # Sort by the number of zeros in corresponding row and column (ascending)
    # This prioritizes positions with fewer alternative zero options
    zero_positions.sort(key=lambda x: x[2])
    
    used_rows = set()
    used_cols = set()
    
    # Assign greedily based on the sorted positions
    for i, j, _ in zero_positions:
        if i not in used_rows and j not in used_cols:
            assignment_matrix[i, j] = 1
            used_rows.add(i)
            used_cols.add(j)
    
    return assignment_matrix

def find_min_line_cover(cost_matrix, assignment_matrix):
    """Find minimum number of lines to cover all zeros"""
    n = cost_matrix.shape[0]
    marked_rows = set()
    marked_cols = set()
    
    # Mark all rows with no assignment
    for i in range(n):
        if not any(assignment_matrix[i, :]):
            marked_rows.add(i)
    
    new_marked = True
    while new_marked:
        new_marked = False
        # Mark columns with zeros in marked rows
        for i in marked_rows:
            for j in range(n):
                if cost_matrix[i, j] == 0 and j not in marked_cols:
                    marked_cols.add(j)
                    new_marked = True
        
        # Mark rows with assignments in marked columns
        for j in marked_cols:
            for i in range(n):
                if assignment_matrix[i, j] == 1 and i not in marked_rows:
                    marked_rows.add(i)
                    new_marked = True
    
    # Lines cover all rows not marked and all columns marked
    cover_rows = set(range(n)) - marked_rows
    cover_cols = marked_cols
    
    return cover_rows, cover_cols

if __name__ == "__main__":
    print("Test 1 - square matrix")    
    cost_matrix = np.array([[1.0, 1e6],
                             [5.81573864e-01, 1e6]])
    
    row_indices, col_indices = solve_hungarian_algorithm(cost_matrix)
    print("Row indices:", row_indices)
    print("Column indices:", col_indices)
    
    print("Test 2 - non-square matrix")
    cost_matrix = np.array([[4, 2, 8, 1],
                             [2, 4, 6, 3],
                             [8, 6, 4, 5]])
    row_indices, col_indices = solve_hungarian_algorithm(cost_matrix)
    print("Row indices:", row_indices)
    print("Column indices:", col_indices)