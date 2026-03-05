from typing import List, Tuple, Optional, Set

INPUT_FILE = "zad_input.txt"
OUTPUT_FILE = "zad_output.txt"

def read_input() -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], List[List[int]]]:
    with open(INPUT_FILE, "r") as f:
        grid = f.read().splitlines()
    
    start_positions = set()
    goal_positions = set()

    my_grid = []
    
    for i in range(len(grid)):
        my_grid.append([])
        for j in range(len(grid[i])):
            my_grid[i].append(0 if grid[i][j] != '#' else 1)
            if grid[i][j] == 'S':
                start_positions.add((i, j))
            elif grid[i][j] == 'G':
                goal_positions.add((i, j))
            elif grid[i][j] == 'B':
                start_positions.add((i, j))
                goal_positions.add((i, j))
    
    return start_positions, goal_positions, my_grid

def write_output(path: str) -> None:
    with open(OUTPUT_FILE, "w") as f:
        f.write(path)

def is_valid_move(grid: List[List[int]], position: Tuple[int, int]) -> bool:
    x, y = position
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0


def print_grid(grid: List[List[int]], positions: Set[Tuple[int, int]], goal_positions: Set[Tuple[int, int]]) -> None:
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (i, j) in positions and (i, j) in goal_positions:
                print('B', end='')  # Both start and goal
            elif (i, j) in positions:
                print('S', end='')
            elif (i, j) in goal_positions:
                print('G', end='')
            else:
                print('#' if grid[i][j] == 1 else ' ', end='')
        print()

# Is win 

def main():
    veteran_positions, goal_positions, grid = read_input()
    print("Start positions:", veteran_positions)
    print("Goal positions:", goal_positions)
    print("Grid:")
    for row in grid:
        print(row)

    print_grid(grid, {(1, 1)}, goal_positions)

    # Tutaj należy zaimplementować algorytm A* lub inny algorytm poszukiwania ścieżki
    # i znaleźć najkrótszą ścieżkę z dowolnego startowego punktu do dowolnego punktu docelowego.
    # Następnie należy zapisać wynik w OUTPUT_FILE.
    # Poniżej znajduje się przykładowa ścieżka, którą można zastąpić rzeczywistym wynikiem algorytmu.

    example_path = "RRUUU"  # Przykładowa ścieżka (prawa, prawa, góra, góra, góra)
    write_output(example_path)

if __name__ == "__main__":
    main() 