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
        covered_rows = set()
        covered_cols = set()
        assigned_rows = [i for i in range(n) if 1 in assignment_matrix[i]]
        unassigned_rows = [i for i in range(n) if i not in assigned_rows]
        marked_rows = set(unassigned_rows)
        marked_cols = set()

        changed = True
        while changed:
            changed = False
            for i in marked_rows.copy():
                for j in range(n):
                    if cost_matrix[i][j] == 0 and j not in marked_cols:
                        marked_cols.add(j)
                        changed = True
            for j in marked_cols.copy():
                for i in range(n):
                    if assignment_matrix[i][j] == 1 and i not in marked_rows:
                        marked_rows.add(i)
                        changed = True

        covered_rows = set(range(n)) - marked_rows
        covered_cols = marked_cols

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

def greedy_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedily assign tasks to workers based on the cost matrix.
    
    Args:
        cost_matrix (np.ndarray): The cost matrix for the assignment problem.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The row and column indices of the greedy assignment.
    """
    
    assignment_matrix = np.zeros(cost_matrix.shape)
    used_columns = set()
    
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            if cost_matrix[i, j] == 0 and j not in used_columns:
                assignment_matrix[i, j] = 1
                used_columns.add(j)
                break
    return assignment_matrix
    

# if __name__ == "__main__":
#     print("Test 1 - square matrix")    
#     cost_matrix = np.array([[4, 2, 8],
#                              [2, 4, 6],
#                              [8, 6, 4]])
    
#     row_indices, col_indices = solve_hungarian_algorithm(cost_matrix)
#     print("Row indices:", row_indices)
#     print("Column indices:", col_indices)
    
#     print("Test 2 - non-square matrix")
#     cost_matrix = np.array([[4, 2, 8, 1],
#                              [2, 4, 6, 3],
#                              [8, 6, 4, 5]])
#     row_indices, col_indices = solve_hungarian_algorithm(cost_matrix)
#     print("Row indices:", row_indices)
#     print("Column indices:", col_indices)