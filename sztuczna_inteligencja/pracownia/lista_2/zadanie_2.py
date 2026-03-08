# Win is when positions - goal_positions is empty, meaning all positions are in goal_positions
# Załóżmy, że niepewność to liczba pozycji, w których może znajdować się weteran

from typing import List, Tuple, Optional, Set
from collections import deque
import random

INPUT_FILE = "zad_input.txt"
OUTPUT_FILE = "zad_output.txt"
UNCERTAINTY = 4

MOVES = {'R': (0, 1), 'L': (0, -1), 'D': (1, 0), 'U': (-1, 0)}  # Right, Left, Down, Up

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

def apply_move(positions: Set[Tuple[int, int]], move: str, grid: List[List[int]]) -> Set[Tuple[int, int]]:
    new_positions = set()
    for pos in positions:
        new_pos = (pos[0] + MOVES[move][0], pos[1] + MOVES[move][1])
        if is_valid_move(grid, new_pos):
            new_positions.add(new_pos)
        else:
            new_positions.add(pos)
    return new_positions

def best_move(positions: Set[Tuple[int, int]], grid: List[List[int]]) -> str:
    best_size = None 
    best_moves = []
    for move in MOVES:
        new_size = len(apply_move(positions, move, grid))
        if best_size is None or new_size < best_size:
            best_size = new_size
            best_moves = [move]
        elif new_size == best_size:
            best_moves.append(move)
    return random.choice(best_moves)

# Tu implemenujemyh zwyczajnego BFSa
def solution(start_positions: Set[Tuple[int, int]], goal_positions: Set[Tuple[int, int]], grid: List[List[int]]) -> Optional[str]:
    path = ""
    # Robienie ruchów zmniejszających niepewność
    while len(start_positions) > UNCERTAINTY:
        move = best_move(start_positions, grid)
        path += move
        start_positions = apply_move(start_positions, move, grid)



    # Zwykłe rozwiązanie BFS, które sprawdza wszystkie możliwe ruchy i dodaje je do kolejki, jeśli są ważne
    queue = deque([(start_positions, path)])  # (current positions, path)
    visited = set()
    visited.add(frozenset(start_positions))

    while queue:
        current_positions, path = queue.popleft()

        if len(current_positions.difference(goal_positions)) == 0:
            return path  # Found a valid path

        for move in MOVES:
            new_positions = set()
            for pos in current_positions:
                new_pos = (pos[0] + MOVES[move][0], pos[1] + MOVES[move][1])
                if is_valid_move(grid, new_pos):
                    new_positions.add(new_pos)
                else:
                    new_positions.add(pos)  # If move is invalid, stay in place
            
            new_positions_frozen = frozenset(new_positions)
            if new_positions_frozen not in visited:
                visited.add(new_positions_frozen)
                queue.append((new_positions_frozen, path + move))

    return None  # No valid path found



def main():
    veteran_positions, goal_positions, grid = read_input()
    # print("Start positions:", veteran_positions)
    # print("Goal positions:", goal_positions)
    # print("Grid:")
    # for row in grid:
    #     print(row)

    # print_grid(grid, {(3, 1)}, goal_positions)

    # Tutaj należy zaimplementować algorytm A* lub inny algorytm poszukiwania ścieżki
    # i znaleźć najkrótszą ścieżkę z dowolnego startowego punktu do dowolnego punktu docelowego.
    # Następnie należy zapisać wynik w OUTPUT_FILE.
    # Poniżej znajduje się przykładowa ścieżka, którą można zastąpić rzeczywistym wynikiem algorytmu.

    result = solution(veteran_positions, goal_positions, grid)
    if result:
        print("Found path:", result)
        write_output(result)
    else:
        print("No path found.")

if __name__ == "__main__":
    main() 