from collections import deque
import os
import sys


UNKNOWN = -1
EMPTY = 0
FILLED = 1


def parse_clue(line):
    line = line.strip()
    if not line:
        return []
    values = [int(x) for x in line.split()]
    if len(values) == 1 and values[0] == 0:
        return []
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
    row_clues = []
    col_clues = []

    idx = 1
    for _ in range(n):
        row_clues.append(parse_clue(lines[idx] if idx < len(lines) else ""))
        idx += 1
    for _ in range(m):
        col_clues.append(parse_clue(lines[idx] if idx < len(lines) else ""))
        idx += 1

    return n, m, row_clues, col_clues, use_files


def build_prefix_tables(line):
    length = len(line)
    can_white = [cell != FILLED for cell in line]
    can_black = [cell != EMPTY for cell in line]

    white_blocked_prefix = [0] * (length + 1)
    black_blocked_prefix = [0] * (length + 1)
    for i in range(length):
        white_blocked_prefix[i + 1] = white_blocked_prefix[i] + (0 if can_white[i] else 1)
        black_blocked_prefix[i + 1] = black_blocked_prefix[i] + (0 if can_black[i] else 1)

    return can_white, can_black, white_blocked_prefix, black_blocked_prefix


def segment_allows_black(black_blocked_prefix, start, end):
    return black_blocked_prefix[end] == black_blocked_prefix[start]


def infer_line(line, clues):
    length = len(line)
    blocks = len(clues)
    can_white, can_black, white_blocked_prefix, black_blocked_prefix = build_prefix_tables(line)

    pref = [[False] * (blocks + 1) for _ in range(length + 1)]
    pref[0][0] = True

    for i in range(1, length + 1):
        for used in range(blocks + 1):
            if can_white[i - 1] and pref[i - 1][used]:
                pref[i][used] = True

            if used == 0:
                continue

            block_len = clues[used - 1]
            start = i - block_len
            if start < 0 or not segment_allows_black(black_blocked_prefix, start, i):
                continue

            if used == 1:
                if pref[start][0]:
                    pref[i][used] = True
            elif start > 0 and can_white[start - 1] and pref[start - 1][used - 1]:
                pref[i][used] = True

    if not pref[length][blocks]:
        raise ValueError("Line has no valid completion")

    suf = [[False] * (blocks + 1) for _ in range(length + 1)]
    suf[length][blocks] = True

    for i in range(length - 1, -1, -1):
        for used in range(blocks, -1, -1):
            if can_white[i] and suf[i + 1][used]:
                suf[i][used] = True

            if used < blocks:
                block_len = clues[used]
                end = i + block_len
                if end <= length and segment_allows_black(black_blocked_prefix, i, end):
                    if used == blocks - 1:
                        if suf[end][used + 1]:
                            suf[i][used] = True
                    elif end < length and can_white[end] and suf[end + 1][used + 1]:
                        suf[i][used] = True

    white_possible = [False] * length
    black_diff = [0] * (length + 1)

    for pos in range(length):
        if can_white[pos]:
            for used in range(blocks + 1):
                if pref[pos][used] and suf[pos + 1][used]:
                    white_possible[pos] = True
                    break

    for block_idx, block_len in enumerate(clues):
        for start in range(length - block_len + 1):
            end = start + block_len
            if not segment_allows_black(black_blocked_prefix, start, end):
                continue

            if block_idx == 0:
                left_ok = pref[start][0]
            else:
                left_ok = start > 0 and can_white[start - 1] and pref[start - 1][block_idx]
            if not left_ok:
                continue

            if block_idx == blocks - 1:
                right_ok = suf[end][block_idx + 1]
            else:
                right_ok = end < length and can_white[end] and suf[end + 1][block_idx + 1]
            if not right_ok:
                continue

            black_diff[start] += 1
            black_diff[end] -= 1

    forced = list(line)
    active = 0
    for pos in range(length):
        active += black_diff[pos]
        black_possible = active > 0
        if not white_possible[pos] and not black_possible:
            raise ValueError("Cell has no valid value")
        if white_possible[pos] and black_possible:
            forced[pos] = UNKNOWN
        elif black_possible:
            forced[pos] = FILLED
        else:
            forced[pos] = EMPTY

    return forced


def solve_nonogram(n, m, row_clues, col_clues):
    board = [[UNKNOWN] * m for _ in range(n)]

    rows_in_queue = [True] * n
    cols_in_queue = [True] * m
    queue = deque(("row", i) for i in range(n))
    queue.extend(("col", j) for j in range(m))

    while queue:
        kind, idx = queue.popleft()
        if kind == "row":
            rows_in_queue[idx] = False
            updated = infer_line(board[idx], row_clues[idx])
            for j, value in enumerate(updated):
                if value == UNKNOWN or board[idx][j] == value:
                    continue
                board[idx][j] = value
                if not cols_in_queue[j]:
                    cols_in_queue[j] = True
                    queue.append(("col", j))
        else:
            cols_in_queue[idx] = False
            column = [board[i][idx] for i in range(n)]
            updated = infer_line(column, col_clues[idx])
            for i, value in enumerate(updated):
                if value == UNKNOWN or board[i][idx] == value:
                    continue
                board[i][idx] = value
                if not rows_in_queue[i]:
                    rows_in_queue[i] = True
                    queue.append(("row", i))

    for row in board:
        if any(cell == UNKNOWN for cell in row):
            raise ValueError("Inference did not finish the board")

    return board


def board_to_output(board):
    return "\n".join(
        "".join("#" if cell == FILLED else "." for cell in row)
        for row in board
    ) + "\n"


def main():
    n, m, row_clues, col_clues, use_files = read_input()
    if n == 0 and m == 0:
        result = ""
    else:
        board = solve_nonogram(n, m, row_clues, col_clues)
        result = board_to_output(board)

    if use_files:
        with open("zad_output.txt", "w", encoding="utf-8") as f:
            f.write(result)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
