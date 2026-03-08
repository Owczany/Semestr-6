from typing import List, Tuple, Optional, Set, Dict
from collections import deque
from itertools import product

INPUT_FILE = "zad_input.txt"
OUTPUT_FILE = "zad_output.txt"
MAX_TOTAL_LEN = 150
TARGET_UNCERTAINTY = 6

MOVES = {
    'R': (0, 1),
    'L': (0, -1),
    'D': (1, 0),
    'U': (-1, 0),
}


def read_input() -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], List[List[int]]]:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    start_positions = set()
    goal_positions = set()
    grid = []

    for i, line in enumerate(lines):
        row = []
        for j, ch in enumerate(line):
            if ch == '#':
                row.append(1)
            else:
                row.append(0)

            if ch == 'S':
                start_positions.add((i, j))
            elif ch == 'G':
                goal_positions.add((i, j))
            elif ch == 'B':
                start_positions.add((i, j))
                goal_positions.add((i, j))
        grid.append(row)

    return start_positions, goal_positions, grid


def write_output(path: str) -> None:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(path)


def is_valid_move(grid: List[List[int]], pos: Tuple[int, int]) -> bool:
    x, y = pos
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0


def move_positions(
    positions: Set[Tuple[int, int]],
    move: str,
    grid: List[List[int]]
) -> Set[Tuple[int, int]]:
    dx, dy = MOVES[move]
    new_positions = set()

    for x, y in positions:
        nx, ny = x + dx, y + dy
        if is_valid_move(grid, (nx, ny)):
            new_positions.add((nx, ny))
        else:
            new_positions.add((x, y))

    return new_positions


def apply_sequence(
    positions: Set[Tuple[int, int]],
    seq: str,
    grid: List[List[int]]
) -> Set[Tuple[int, int]]:
    current = set(positions)
    for mv in seq:
        current = move_positions(current, mv, grid)
    return current


def compute_goal_distances(
    grid: List[List[int]],
    goal_positions: Set[Tuple[int, int]]
) -> Dict[Tuple[int, int], int]:
    """
    Multi-source BFS od wszystkich pól docelowych.
    Przydaje się do tie-breaka w fazie zachłannej.
    """
    dist = {}
    q = deque()

    for g in goal_positions:
        dist[g] = 0
        q.append(g)

    while q:
        x, y = q.popleft()
        for dx, dy in MOVES.values():
            nx, ny = x + dx, y + dy
            if is_valid_move(grid, (nx, ny)) and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))

    return dist


def state_score(
    positions: Set[Tuple[int, int]],
    goal_dist: Dict[Tuple[int, int], int]
) -> Tuple[int, int]:
    """
    Mniejszy score jest lepszy:
    1. mniej możliwych pozycji,
    2. mniejsza suma odległości do najbliższego celu.
    """
    BIG = 10**9
    return (
        len(positions),
        sum(goal_dist.get(p, BIG) for p in positions)
    )


def greedy_reduce_uncertainty(
    start_positions: Set[Tuple[int, int]],
    goal_positions: Set[Tuple[int, int]],
    grid: List[List[int]],
    target_uncertainty: int,
    max_prefix_len: int
) -> Tuple[Set[Tuple[int, int]], str]:
    """
    Faza 1:
    Zachłannie wybieramy krótką sekwencję ruchów (długości 1..3),
    która najlepiej zmniejsza niepewność.
    """
    goal_dist = compute_goal_distances(grid, goal_positions)
    current = set(start_positions)
    path = ""

    candidate_sequences = []

    # wszystkie sekwencje długości 1, 2, 3
    for length in [1, 2, 3]:
        for seq in product("UDLR", repeat=length):
            candidate_sequences.append("".join(seq))

    while len(current) > target_uncertainty and len(path) < max_prefix_len:
        best_seq = None
        best_positions = None
        best_score = state_score(current, goal_dist)

        for seq in candidate_sequences:
            if len(path) + len(seq) > max_prefix_len:
                continue

            new_positions = apply_sequence(current, seq, grid)
            sc = state_score(new_positions, goal_dist)

            if sc < best_score:
                best_score = sc
                best_seq = seq
                best_positions = new_positions

        if best_seq is None:
            # Nie udało się znaleźć nic lepszego - kończymy fazę 1
            break

        path += best_seq
        current = best_positions

    return current, path


def bfs_from_state(
    start_state: Set[Tuple[int, int]],
    goal_positions: Set[Tuple[int, int]],
    grid: List[List[int]],
    max_depth: int
) -> Optional[str]:
    """
    BFS po stanach typu: zbiór możliwych pozycji komandosa.
    """
    start_frozen = frozenset(start_state)

    if start_frozen.issubset(goal_positions):
        return ""

    queue = deque([(start_frozen, "")])
    visited = {start_frozen}

    while queue:
        current_state, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for move in "UDLR":
            new_positions = move_positions(set(current_state), move, grid)
            new_state = frozenset(new_positions)

            if new_state in visited:
                continue

            new_path = path + move

            if new_state.issubset(goal_positions):
                return new_path

            visited.add(new_state)
            queue.append((new_state, new_path))

    return None


def solution(
    start_positions: Set[Tuple[int, int]],
    goal_positions: Set[Tuple[int, int]],
    grid: List[List[int]]
) -> Optional[str]:
    # Faza 1 - redukcja niepewności
    reduced_positions, prefix = greedy_reduce_uncertainty(
        start_positions=start_positions,
        goal_positions=goal_positions,
        grid=grid,
        target_uncertainty=TARGET_UNCERTAINTY,
        max_prefix_len=80
    )

    # Jeśli już wygraliśmy po fazie 1
    if reduced_positions.issubset(goal_positions):
        return prefix if len(prefix) < MAX_TOTAL_LEN else None

    remaining_len = MAX_TOTAL_LEN - len(prefix)
    if remaining_len < 0:
        return None

    # Faza 2 - BFS
    suffix = bfs_from_state(
        start_state=reduced_positions,
        goal_positions=goal_positions,
        grid=grid,
        max_depth=remaining_len
    )

    if suffix is None:
        return None

    result = prefix + suffix
    if len(result) >= MAX_TOTAL_LEN:
        return None

    return result


def main():
    start_positions, goal_positions, grid = read_input()
    result = solution(start_positions, goal_positions, grid)

    if result is not None:
        write_output(result)
        print("Found path:", result)
    else:
        print("No path found.")


if __name__ == "__main__":
    main()