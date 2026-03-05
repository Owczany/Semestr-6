#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import lru_cache
from typing import List, Tuple, Optional

INPUT_FILE = "zad_input.txt"
OUTPUT_FILE = "zad_output.txt"

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

def write_solution(path: str, grid: List[List[int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in grid:
            f.write("".join("#" if v else "." for v in row) + "\n")

# -----------------------------
# Generate all patterns for a line
# -----------------------------

@lru_cache(maxsize=None)
def generate_patterns(n: int, clues: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    """
    Zwraca wszystkie poprawne wzorce 0/1 długości n pasujące do clues.
    clues = (c1, c2, ..., ck) oznacza k bloków jedynek, między blokami >=1 zero.
    """
    if len(clues) == 0:
        return (tuple([0] * n),)

    k = len(clues)
    total_ones = sum(clues)
    min_gaps = k - 1
    min_len = total_ones + min_gaps
    if min_len > n:
        return tuple()  # sprzeczne

    res = []

    # rekurencyjnie ustawiamy bloki
    def rec(idx: int, pos: int, arr: List[int]):
        if idx == k:
            res.append(tuple(arr))
            return

        # minimalna długość potrzebna na bloki idx..k-1 + przerwy między nimi
        remaining_ones = sum(clues[idx:])
        remaining_gaps = (k - 1 - idx)
        min_remaining = remaining_ones + remaining_gaps

        block_len = clues[idx]

        # start bieżącego bloku
        for s in range(pos, n - min_remaining + 1):
            arr2 = arr[:]  # copy
            # wstaw blok
            for t in range(s, s + block_len):
                arr2[t] = 1

            next_pos = s + block_len
            if idx < k - 1:
                # wymuś co najmniej jedno zero po bloku
                if next_pos >= n:
                    continue
                arr2[next_pos] = 0
                next_pos += 1

            rec(idx + 1, next_pos, arr2)

    rec(0, 0, [0] * n)
    return tuple(res)

# -----------------------------
# Constraint propagation + backtracking
# -----------------------------

def intersect_forced_bits(patterns: List[Tuple[int, ...]]) -> List[Optional[int]]:
    """
    Dla listy wzorców zwraca listę długości n:
    - 0 jeśli na tej pozycji wszystkie wzorce mają 0
    - 1 jeśli wszystkie mają 1
    - None jeśli różnie
    """
    n = len(patterns[0])
    forced = [None] * n
    for j in range(n):
        v = patterns[0][j]
        same = True
        for p in patterns[1:]:
            if p[j] != v:
                same = False
                break
        if same:
            forced[j] = v
    return forced

def filter_patterns_by_bit(patterns: List[Tuple[int, ...]], idx: int, val: int) -> List[Tuple[int, ...]]:
    return [p for p in patterns if p[idx] == val]

def propagate(
    X: int, Y: int,
    row_pats: List[List[Tuple[int, ...]]],
    col_pats: List[List[Tuple[int, ...]]],
    grid: List[List[Optional[int]]],
) -> bool:
    """
    Propagacja ograniczeń.
    Zwraca False jeśli wykryto sprzeczność (pusta domena).
    """
    changed = True
    while changed:
        changed = False

        # 1) z wierszy narzucamy kolumnom
        for i in range(X):
            if not row_pats[i]:
                return False
            forced = intersect_forced_bits(row_pats[i])
            for j, fv in enumerate(forced):
                if fv is None:
                    continue
                if grid[i][j] is None:
                    grid[i][j] = fv
                    # odetnij w kolumnie wzorce niepasujące
                    newc = filter_patterns_by_bit(col_pats[j], i, fv)
                    if len(newc) != len(col_pats[j]):
                        col_pats[j] = newc
                        changed = True
                    if not col_pats[j]:
                        return False
                    changed = True
                else:
                    if grid[i][j] != fv:
                        return False

        # 2) z kolumn narzucamy wierszom
        for j in range(Y):
            if not col_pats[j]:
                return False
            forced = intersect_forced_bits(col_pats[j])
            for i, fv in enumerate(forced):
                if fv is None:
                    continue
                if grid[i][j] is None:
                    grid[i][j] = fv
                    newr = filter_patterns_by_bit(row_pats[i], j, fv)
                    if len(newr) != len(row_pats[i]):
                        row_pats[i] = newr
                        changed = True
                    if not row_pats[i]:
                        return False
                    changed = True
                else:
                    if grid[i][j] != fv:
                        return False

        # 3) dodatkowe czyszczenie domen po znanych komórkach
        # (czasem po narzuceniu wielu pól warto odfiltrować całe wzorce)
        for i in range(X):
            if not row_pats[i]:
                return False
            fixed = grid[i]
            if any(v is not None for v in fixed):
                before = len(row_pats[i])
                row_pats[i] = [p for p in row_pats[i] if all(fixed[j] is None or p[j] == fixed[j] for j in range(Y))]
                if not row_pats[i]:
                    return False
                if len(row_pats[i]) != before:
                    changed = True

        for j in range(Y):
            if not col_pats[j]:
                return False
            fixed_col = [grid[i][j] for i in range(X)]
            if any(v is not None for v in fixed_col):
                before = len(col_pats[j])
                col_pats[j] = [p for p in col_pats[j] if all(fixed_col[i] is None or p[i] == fixed_col[i] for i in range(X))]
                if not col_pats[j]:
                    return False
                if len(col_pats[j]) != before:
                    changed = True

    return True

def is_solved(grid: List[List[Optional[int]]]) -> bool:
    return all(all(v is not None for v in row) for row in grid)

def choose_branch_line(
    row_pats: List[List[Tuple[int, ...]]],
    col_pats: List[List[Tuple[int, ...]]],
) -> Tuple[str, int]:
    """
    Heurystyka: wybierz linię z najmniejszą domeną > 1.
    Zwraca ('r', i) albo ('c', j).
    """
    best_type = 'r'
    best_idx = -1
    best_size = 10**9

    for i, dom in enumerate(row_pats):
        s = len(dom)
        if 1 < s < best_size:
            best_size = s
            best_type = 'r'
            best_idx = i

    for j, dom in enumerate(col_pats):
        s = len(dom)
        if 1 < s < best_size:
            best_size = s
            best_type = 'c'
            best_idx = j

    return best_type, best_idx

def deep_copy_domains(domains: List[List[Tuple[int, ...]]]) -> List[List[Tuple[int, ...]]]:
    return [d[:] for d in domains]

def deep_copy_grid(grid: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
    return [row[:] for row in grid]

def solve_backtracking(
    X: int, Y: int,
    row_pats: List[List[Tuple[int, ...]]],
    col_pats: List[List[Tuple[int, ...]]],
    grid: List[List[Optional[int]]],
) -> Optional[List[List[int]]]:
    # propagacja na start
    if not propagate(X, Y, row_pats, col_pats, grid):
        return None

    if is_solved(grid):
        return [[int(v) for v in row] for row in grid]

    line_type, idx = choose_branch_line(row_pats, col_pats)
    if idx == -1:
        # teoretycznie może się zdarzyć, że domeny mają 1, ale grid nie wypełniony
        # wtedy dopiszmy z domen
        for i in range(X):
            if any(v is None for v in grid[i]):
                p = row_pats[i][0]
                for j in range(Y):
                    grid[i][j] = p[j]
        return [[int(v) for v in row] for row in grid]

    if line_type == 'r':
        domain = row_pats[idx]
        for pat in domain:
            rp2 = deep_copy_domains(row_pats)
            cp2 = deep_copy_domains(col_pats)
            g2 = deep_copy_grid(grid)

            rp2[idx] = [pat]
            # narzuć cały wiersz do grid
            for j in range(Y):
                if g2[idx][j] is None:
                    g2[idx][j] = pat[j]
                elif g2[idx][j] != pat[j]:
                    break
            else:
                ans = solve_backtracking(X, Y, rp2, cp2, g2)
                if ans is not None:
                    return ans
        return None

    else:
        domain = col_pats[idx]
        for pat in domain:
            rp2 = deep_copy_domains(row_pats)
            cp2 = deep_copy_domains(col_pats)
            g2 = deep_copy_grid(grid)

            cp2[idx] = [pat]
            # narzuć całą kolumnę do grid
            for i in range(X):
                if g2[i][idx] is None:
                    g2[i][idx] = pat[i]
                elif g2[i][idx] != pat[i]:
                    break
            else:
                ans = solve_backtracking(X, Y, rp2, cp2, g2)
                if ans is not None:
                    return ans
        return None

def solve_nonogram(X: int, Y: int, row_desc: List[Tuple[int, ...]], col_desc: List[Tuple[int, ...]]) -> List[List[int]]:
    # domeny początkowe
    row_pats = [list(generate_patterns(Y, row_desc[i])) for i in range(X)]
    col_pats = [list(generate_patterns(X, col_desc[j])) for j in range(Y)]

    # pusty grid (None = nieustalone)
    grid: List[List[Optional[int]]] = [[None for _ in range(Y)] for _ in range(X)]

    ans = solve_backtracking(X, Y, row_pats, col_pats, grid)
    if ans is None:
        # validator zakłada, że jest rozwiązanie, ale na wszelki wypadek:
        # zwróć kropki
        return [[0 for _ in range(Y)] for _ in range(X)]
    return ans

# -----------------------------
# Main
# -----------------------------

def main():
    X, Y, row_desc, col_desc = read_instance(INPUT_FILE)
    grid = solve_nonogram(X, Y, row_desc, col_desc)
    write_solution(OUTPUT_FILE, grid)

if __name__ == "__main__":
    main()