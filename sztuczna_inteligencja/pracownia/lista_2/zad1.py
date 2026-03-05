#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from functools import lru_cache
from typing import List, Tuple

INPUT_FILE = "zad_input.txt"
OUTPUT_FILE = "zad_output.txt"
INF = 10**9

# -----------------------------
# Parsing
# -----------------------------

def parse_clue_line(s: str) -> Tuple[int, ...]:
    s = s.strip()
    if s == "":
        return tuple()
    nums = tuple(int(x) for x in s.split())
    if len(nums) == 1 and nums[0] == 0:
        return tuple()
    return nums

def read_instance(path: str) -> Tuple[int, int, List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    X, Y = map(int, lines[0].split())
    row_desc = [parse_clue_line(lines[1 + i]) for i in range(X)]
    col_desc = [parse_clue_line(lines[1 + X + j]) for j in range(Y)]
    return X, Y, row_desc, col_desc

def write_solution(path: str, X: int, Y: int, row_masks: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(X):
            m = row_masks[i]
            # bit j => kolumna j
            line = []
            for j in range(Y):
                line.append("#" if ((m >> j) & 1) else ".")
            f.write("".join(line) + "\n")

# -----------------------------
# Pattern generation (all valid line bitmasks for a clue)
# -----------------------------

@lru_cache(maxsize=None)
def generate_patterns(n: int, clues: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Zwraca wszystkie poprawne wzorce jako bitmaski (int) długości n.
    Bit j=1 oznacza # na pozycji j.
    """
    if len(clues) == 0:
        return (0,)

    k = len(clues)
    total_ones = sum(clues)
    min_gaps = k - 1
    if total_ones + min_gaps > n:
        return tuple()  # sprzeczne

    res: List[int] = []

    def rec(idx: int, pos: int, mask: int):
        if idx == k:
            res.append(mask)
            return

        remaining_ones = sum(clues[idx:])
        remaining_gaps = (k - 1 - idx)
        min_remaining = remaining_ones + remaining_gaps
        block_len = clues[idx]

        for s in range(pos, n - min_remaining + 1):
            m2 = mask
            # ustaw blok jedynek
            for t in range(s, s + block_len):
                m2 |= (1 << t)

            next_pos = s + block_len
            if idx < k - 1:
                # wymuś co najmniej jedno zero po bloku => po prostu przesuwamy pos o 1
                if next_pos >= n:
                    continue
                next_pos += 1

            rec(idx + 1, next_pos, m2)

    rec(0, 0, 0)
    return tuple(res)

def line_cost(mask: int, patterns: Tuple[int, ...]) -> int:
    """
    Minimalna liczba flipów żeby mask pasował do clue (czyli do jednego z patterns).
    """
    best = INF
    for p in patterns:
        d = (mask ^ p).bit_count()
        if d < best:
            best = d
            if best == 0:
                break
    return best

# -----------------------------
# WalkSat-like solver
# -----------------------------

def random_grid_masks(X: int, Y: int, rng: random.Random, p: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Losowy grid w bitmaskach:
    - row_masks[i]: Y-bit
    - col_masks[j]: X-bit
    """
    row_masks = [0] * X
    col_masks = [0] * Y
    for i in range(X):
        m = 0
        for j in range(Y):
            if rng.random() < p:
                m |= (1 << j)
                col_masks[j] |= (1 << i)
        row_masks[i] = m
    return row_masks, col_masks

def solve_nonogram_walksat(
    X: int, Y: int,
    row_desc: List[Tuple[int, ...]],
    col_desc: List[Tuple[int, ...]],
    seed: int = 42,
    max_steps: int = 400_000,
    restarts: int = 80,
    restart_after_no_improve: int = 6000,
    p_one: float = 0.5,
    noise_random_flip: float = 0.03,
    noise_line_only: float = 0.05,
):
    """
    WalkSat-like:
    - start z losowej planszy
    - wybieraj losowo złą linię
    - flipuj komórkę minimalizując koszt (wiersz+kolumna), czasem szum
    - restart gdy brak poprawy
    """
    rng = random.Random(seed)

    # precompute patterns per row/col
    row_patterns = [generate_patterns(Y, row_desc[i]) for i in range(X)]
    col_patterns = [generate_patterns(X, col_desc[j]) for j in range(Y)]

    # szybka detekcja sprzecznych opisów
    if any(len(p) == 0 for p in row_patterns) or any(len(p) == 0 for p in col_patterns):
        # sprzeczna instancja (nie powinna w testach)
        row_masks = [0] * X
        return row_masks

    best_row_masks = None
    best_total = INF

    steps_used = 0

    for _r in range(restarts):
        row_masks, col_masks = random_grid_masks(X, Y, rng, p=p_one)

        row_costs = [line_cost(row_masks[i], row_patterns[i]) for i in range(X)]
        col_costs = [line_cost(col_masks[j], col_patterns[j]) for j in range(Y)]
        total = sum(row_costs) + sum(col_costs)

        if total < best_total:
            best_total = total
            best_row_masks = row_masks[:]
            if best_total == 0:
                return best_row_masks

        last_improve = steps_used

        while steps_used < max_steps:
            steps_used += 1

            if total == 0:
                return row_masks

            # plateau restart
            if steps_used - last_improve >= restart_after_no_improve:
                break

            bad_rows = [i for i in range(X) if row_costs[i] > 0]
            bad_cols = [j for j in range(Y) if col_costs[j] > 0]

            if not bad_rows and not bad_cols:
                return row_masks

            # wybór linii: proporcjonalnie do liczby złych
            if bad_rows and bad_cols:
                choose_row = (rng.random() < (len(bad_rows) / (len(bad_rows) + len(bad_cols))))
            else:
                choose_row = bool(bad_rows)

            # szum: losowy flip gdziekolwiek
            if rng.random() < noise_random_flip:
                i = rng.randrange(X)
                j = rng.randrange(Y)
            else:
                if choose_row:
                    i = rng.choice(bad_rows)

                    # czasem optymalizuj tylko wiersz (ignoruj kolumnę)
                    row_only = (rng.random() < noise_line_only)

                    best_js = []
                    best_delta = None

                    old_row_mask = row_masks[i]
                    old_row_cost = row_costs[i]

                    for j in range(Y):
                        new_row_mask = old_row_mask ^ (1 << j)
                        new_row_cost = line_cost(new_row_mask, row_patterns[i])

                        if row_only:
                            new_col_cost = col_costs[j]
                        else:
                            old_col_mask = col_masks[j]
                            new_col_mask = old_col_mask ^ (1 << i)
                            new_col_cost = line_cost(new_col_mask, col_patterns[j])

                        delta = (new_row_cost - old_row_cost) + (new_col_cost - col_costs[j])
                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            best_js = [j]
                        elif delta == best_delta:
                            best_js.append(j)

                    j = rng.choice(best_js)

                else:
                    j = rng.choice(bad_cols)

                    col_only = (rng.random() < noise_line_only)

                    best_is = []
                    best_delta = None

                    old_col_mask = col_masks[j]
                    old_col_cost = col_costs[j]

                    for i in range(X):
                        new_col_mask = old_col_mask ^ (1 << i)
                        new_col_cost = line_cost(new_col_mask, col_patterns[j])

                        if col_only:
                            new_row_cost = row_costs[i]
                        else:
                            old_row_mask = row_masks[i]
                            new_row_mask = old_row_mask ^ (1 << j)
                            new_row_cost = line_cost(new_row_mask, row_patterns[i])

                        delta = (new_row_cost - row_costs[i]) + (new_col_cost - old_col_cost)
                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            best_is = [i]
                        elif delta == best_delta:
                            best_is.append(i)

                    i = rng.choice(best_is)

            # wykonaj flip (i,j) + aktualizacja masek
            row_masks[i] ^= (1 << j)
            col_masks[j] ^= (1 << i)

            # aktualizacja kosztów tylko dla tego wiersza i tej kolumny
            row_costs[i] = line_cost(row_masks[i], row_patterns[i])
            col_costs[j] = line_cost(col_masks[j], col_patterns[j])

            new_total = sum(row_costs) + sum(col_costs)
            total = new_total

            if total < best_total:
                best_total = total
                best_row_masks = row_masks[:]
                last_improve = steps_used
                if best_total == 0:
                    return best_row_masks

    # jak nie trafiliśmy 0, zwróć najlepsze co było (validator i tak zaliczy tylko 0)
    return best_row_masks if best_row_masks is not None else [0] * X

# -----------------------------
# Main
# -----------------------------

def main():
    X, Y, row_desc, col_desc = read_instance(INPUT_FILE)
    row_masks = solve_nonogram_walksat(
        X, Y, row_desc, col_desc,
        seed=42,
        max_steps=450_000,
        restarts=100,
        restart_after_no_improve=7000,
        p_one=0.5,
        noise_random_flip=0.03,
        noise_line_only=0.05,
    )
    write_solution(OUTPUT_FILE, X, Y, row_masks)

if __name__ == "__main__":
    main()