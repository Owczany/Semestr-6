from collections import deque
from functools import lru_cache
import os
import sys


def parse_clue(line):
    line = line.strip()
    if not line:
        return ()
    values = tuple(int(x) for x in line.split())
    if len(values) == 1 and values[0] == 0:
        return ()
    return values


def read_input():
    if os.path.exists("zad_input.txt"):
        with open("zad_input.txt", "r", encoding="utf-8") as f:
            data = f.read()
        use_files = True
    else:
        data = sys.stdin.read()
        use_files = False

    lines = data.splitlines()
    if not lines:
        return 0, 0, [], [], use_files

    n, m = map(int, lines[0].split())
    idx = 1
    row_clues = []
    col_clues = []

    for _ in range(n):
        row_clues.append(parse_clue(lines[idx] if idx < len(lines) else ""))
        idx += 1
    for _ in range(m):
        col_clues.append(parse_clue(lines[idx] if idx < len(lines) else ""))
        idx += 1

    return n, m, row_clues, col_clues, use_files


@lru_cache(maxsize=None)
def generate_patterns(length, clues):
    if not clues:
        return (0,)

    suffix = [0] * (len(clues) + 1)
    for i in range(len(clues) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + clues[i]

    patterns = []

    def rec(block_idx, pos, mask):
        if block_idx == len(clues):
            patterns.append(mask)
            return

        block_len = clues[block_idx]
        remaining_blocks = len(clues) - block_idx - 1
        min_rest = suffix[block_idx + 1] + remaining_blocks
        max_start = length - block_len - min_rest

        for start in range(pos, max_start + 1):
            block_mask = ((1 << block_len) - 1) << start
            next_pos = start + block_len + 1
            rec(block_idx + 1, next_pos, mask | block_mask)

    rec(0, 0, 0)
    return tuple(patterns)


def filter_patterns(patterns, required_one, required_zero):
    return [mask for mask in patterns if (mask & required_one) == required_one and (mask & required_zero) == 0]


def forced_masks(patterns, full_mask):
    common_one = full_mask
    any_one = 0
    for mask in patterns:
        common_one &= mask
        any_one |= mask
    common_zero = full_mask ^ any_one
    return common_one, common_zero


def collect_row_requirements(state, row_idx):
    required_one = 0
    required_zero = 0
    for col_idx, mask in enumerate(state["col_known_one"]):
        if (mask >> row_idx) & 1:
            required_one |= 1 << col_idx
    for col_idx, mask in enumerate(state["col_known_zero"]):
        if (mask >> row_idx) & 1:
            required_zero |= 1 << col_idx
    return required_one, required_zero


def collect_col_requirements(state, col_idx):
    required_one = 0
    required_zero = 0
    for row_idx, mask in enumerate(state["row_known_one"]):
        if (mask >> col_idx) & 1:
            required_one |= 1 << row_idx
    for row_idx, mask in enumerate(state["row_known_zero"]):
        if (mask >> col_idx) & 1:
            required_zero |= 1 << row_idx
    return required_one, required_zero


def enqueue(queue, in_queue, kind, idx):
    key = (kind, idx)
    if key not in in_queue:
        in_queue.add(key)
        queue.append(key)


def propagate(state):
    n = state["n"]
    m = state["m"]
    row_full = state["row_full"]
    col_full = state["col_full"]

    queue = deque()
    in_queue = set()
    for i in range(n):
        enqueue(queue, in_queue, "row", i)
    for j in range(m):
        enqueue(queue, in_queue, "col", j)

    while queue:
        kind, idx = queue.popleft()
        in_queue.discard((kind, idx))

        if kind == "row":
            required_one, required_zero = collect_row_requirements(state, idx)
            candidates = filter_patterns(state["row_patterns"][idx], required_one, required_zero)
            if not candidates:
                return False
            if len(candidates) != len(state["row_patterns"][idx]):
                state["row_patterns"][idx] = candidates

            new_one, new_zero = forced_masks(candidates, row_full)
            if new_one & new_zero:
                return False

            changed = (new_one ^ state["row_known_one"][idx]) | (new_zero ^ state["row_known_zero"][idx])
            if changed:
                state["row_known_one"][idx] = new_one
                state["row_known_zero"][idx] = new_zero
                for col_idx in range(m):
                    if (changed >> col_idx) & 1:
                        enqueue(queue, in_queue, "col", col_idx)

        else:
            required_one, required_zero = collect_col_requirements(state, idx)
            candidates = filter_patterns(state["col_patterns"][idx], required_one, required_zero)
            if not candidates:
                return False
            if len(candidates) != len(state["col_patterns"][idx]):
                state["col_patterns"][idx] = candidates

            new_one, new_zero = forced_masks(candidates, col_full)
            if new_one & new_zero:
                return False

            changed = (new_one ^ state["col_known_one"][idx]) | (new_zero ^ state["col_known_zero"][idx])
            if changed:
                state["col_known_one"][idx] = new_one
                state["col_known_zero"][idx] = new_zero
                for row_idx in range(n):
                    if (changed >> row_idx) & 1:
                        enqueue(queue, in_queue, "row", row_idx)

    return True


def is_solved(state):
    return all(len(patterns) == 1 for patterns in state["row_patterns"])


def choose_branch_line(state):
    best = None

    for i, patterns in enumerate(state["row_patterns"]):
        size = len(patterns)
        if 1 < size and (best is None or size < best[2]):
            best = ("row", i, size)

    for j, patterns in enumerate(state["col_patterns"]):
        size = len(patterns)
        if 1 < size and (best is None or size < best[2]):
            best = ("col", j, size)

    return best


def clone_state(state):
    return {
        "n": state["n"],
        "m": state["m"],
        "row_full": state["row_full"],
        "col_full": state["col_full"],
        "row_patterns": [patterns[:] for patterns in state["row_patterns"]],
        "col_patterns": [patterns[:] for patterns in state["col_patterns"]],
        "row_known_one": state["row_known_one"][:],
        "row_known_zero": state["row_known_zero"][:],
        "col_known_one": state["col_known_one"][:],
        "col_known_zero": state["col_known_zero"][:],
    }


def solve(state):
    if not propagate(state):
        return None

    if is_solved(state):
        return state

    branch = choose_branch_line(state)
    if branch is None:
        return None

    kind, idx, _ = branch
    options = state["row_patterns"][idx] if kind == "row" else state["col_patterns"][idx]

    for mask in options:
        next_state = clone_state(state)
        if kind == "row":
            next_state["row_patterns"][idx] = [mask]
        else:
            next_state["col_patterns"][idx] = [mask]

        result = solve(next_state)
        if result is not None:
            return result

    return None


def build_initial_state(n, m, row_clues, col_clues):
    return {
        "n": n,
        "m": m,
        "row_full": (1 << m) - 1,
        "col_full": (1 << n) - 1,
        "row_patterns": [list(generate_patterns(m, clue)) for clue in row_clues],
        "col_patterns": [list(generate_patterns(n, clue)) for clue in col_clues],
        "row_known_one": [0] * n,
        "row_known_zero": [0] * n,
        "col_known_one": [0] * m,
        "col_known_zero": [0] * m,
    }


def row_mask_to_string(mask, width):
    return "".join("#" if (mask >> j) & 1 else "." for j in range(width))


def solve_nonogram(n, m, row_clues, col_clues):
    state = build_initial_state(n, m, row_clues, col_clues)
    solved = solve(state)
    if solved is None:
        raise ValueError("Puzzle has no solution")
    return [patterns[0] for patterns in solved["row_patterns"]]


def main():
    n, m, row_clues, col_clues, use_files = read_input()
    if n == 0 and m == 0:
        result = ""
    else:
        row_masks = solve_nonogram(n, m, row_clues, col_clues)
        result = "\n".join(row_mask_to_string(mask, m) for mask in row_masks) + "\n"

    if use_files:
        with open("zad_output.txt", "w", encoding="utf-8") as f:
            f.write(result)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
