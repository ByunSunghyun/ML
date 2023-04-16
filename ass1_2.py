import numpy as np

def euclidean_norm(matrix):
    return np.linalg.norm(matrix)

# Example
matrix = np.array([[1, 2, 3, 4], [1, 8, 1, 8], [10, 6, 4, 2], [0, 2, 4, 6]])

norm = euclidean_norm(matrix)
print(norm)