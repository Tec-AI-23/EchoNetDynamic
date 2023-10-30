def pixel_expand(matrix, x, y, expansion=3):
    print(matrix[x][y])
    if matrix[x][y] != 1:
        print("wut")
        return matrix

    rows, cols = len(matrix), len(matrix[0])

    def dfs(row, col):
        if row < 0 or row >= rows or col < 0 or col >= cols or matrix[row][col] != 0:
            return
        matrix[row][col] = 1  # Expand the circle
        dfs(row - 1, col)  # Up
        dfs(row + 1, col)  # Down
        dfs(row, col - 1)  # Left
        dfs(row, col + 1)  # Right

        dfs(x, y)

    return matrix


# Example usage:
matrix = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

expanded_matrix = pixel_expand(matrix, 2, 2)

# Print the expanded matrix
for row in expanded_matrix:
    print(row)
