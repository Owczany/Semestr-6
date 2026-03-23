from collections import deque
from itertools import product
import os
import random
import sys


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_CHARS = "UDLR"


def iter_bits(mask: int):
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


def popcount(x: int) -> int:
    return x.bit_count()


def read_input():
    if os.path.exists("zad_input.txt"):
        with open("zad_input.txt", "r", encoding="utf-8") as f:
            data = f.read().splitlines()
    else:
        data = sys.stdin.read().splitlines()
    while data and data[-1] == "":
        data.pop()
    return data


def write_output(ans: str):
    with open("zad_output.txt", "w", encoding="utf-8") as f:
        f.write(ans + "\n")


class Solver:
    def __init__(self, lines):
        self.grid = lines
        self.h = len(lines)
        self.w = len(lines[0]) if self.h else 0

        self.pos_to_id = {}
        self.id_to_pos = []
        self.starts = []
        self.goals = []

        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r][c] != '#':
                    idx = len(self.id_to_pos)
                    self.pos_to_id[(r, c)] = idx
                    self.id_to_pos.append((r, c))

        self.n = len(self.id_to_pos)

        for idx, (r, c) in enumerate(self.id_to_pos):
            ch = self.grid[r][c]
            if ch in ('S', 'B'):
                self.starts.append(idx)
            if ch in ('G', 'B'):
                self.goals.append(idx)

        self.start_mask = 0
        for s in self.starts:
            self.start_mask |= 1 << s

        self.goal_mask = 0
        for g in self.goals:
            self.goal_mask |= 1 << g

        self.next_pos = [[0] * self.n for _ in range(4)]
        for idx, (r, c) in enumerate(self.id_to_pos):
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.pos_to_id:
                    self.next_pos[d][idx] = self.pos_to_id[(nr, nc)]
                else:
                    self.next_pos[d][idx] = idx

        self.dist_to_goal = self._compute_goal_dist()
        self.trans_cache = [dict() for _ in range(4)]

        self.short_seqs = self._build_short_seqs()
        self.long_pushes = self._build_long_pushes()

    def _compute_goal_dist(self):
        INF = 10**9
        dist = [INF] * self.n
        q = deque()

        for g in self.goals:
            dist[g] = 0
            q.append(g)

        while q:
            v = q.popleft()
            vr, vc = self.id_to_pos[v]
            for dr, dc in DIRS:
                nr, nc = vr + dr, vc + dc
                if (nr, nc) in self.pos_to_id:
                    u = self.pos_to_id[(nr, nc)]
                    if dist[u] == INF:
                        dist[u] = dist[v] + 1
                        q.append(u)
        return dist

    def _build_short_seqs(self):
        seqs = []
        for length in (1, 2, 3):
            for tup in product(range(4), repeat=length):
                seqs.append(tup)
        return seqs

    def _build_long_pushes(self):
        U = tuple([0] * self.h)
        D = tuple([1] * self.h)
        L = tuple([2] * self.w)
        R = tuple([3] * self.w)

        pushes = [U, D, L, R]

        seqs = []
        for s in pushes:
            seqs.append(s)

        for a in pushes:
            for b in pushes:
                if a != b:
                    seqs.append(a + b)

        # krótsze bloki
        k = min(6, max(self.h, self.w))
        seqs.extend([
            tuple([0] * min(k, self.h)),
            tuple([1] * min(k, self.h)),
            tuple([2] * min(k, self.w)),
            tuple([3] * min(k, self.w)),
        ])

        return seqs

    def is_goal_state(self, mask: int) -> bool:
        return (mask & ~self.goal_mask) == 0

    def move_once(self, mask: int, d: int) -> int:
        cached = self.trans_cache[d].get(mask)
        if cached is not None:
            return cached

        nxt = 0
        trans = self.next_pos[d]
        for i in iter_bits(mask):
            nxt |= 1 << trans[i]

        self.trans_cache[d][mask] = nxt
        return nxt

    def apply_seq(self, mask: int, seq) -> int:
        for d in seq:
            mask = self.move_once(mask, d)
        return mask

    def score_mask(self, mask: int):
        cnt = popcount(mask)
        max_dist = 0
        dist_sum = 0

        min_r = 10**9
        max_r = -1
        min_c = 10**9
        max_c = -1

        for i in iter_bits(mask):
            r, c = self.id_to_pos[i]
            if r < min_r:
                min_r = r
            if r > max_r:
                max_r = r
            if c < min_c:
                min_c = c
            if c > max_c:
                max_c = c

            d = self.dist_to_goal[i]
            dist_sum += d
            if d > max_dist:
                max_dist = d

        box = (max_r - min_r) + (max_c - min_c)
        return (cnt, box, max_dist, dist_sum)

    def reconstruct_bfs_path(self, parent, end_state):
        path = []
        cur = end_state
        while parent[cur][0] != -1:
            prev, move = parent[cur]
            path.append(DIR_CHARS[move])
            cur = prev
        path.reverse()
        return "".join(path)

    def bounded_bfs(self, start_mask: int, max_depth: int, node_limit: int):
        if self.is_goal_state(start_mask):
            return ""

        q = deque([(start_mask, 0)])
        parent = {start_mask: (-1, -1)}
        visited_nodes = 1

        while q:
            state, dep = q.popleft()
            if dep >= max_depth:
                continue

            for d in range(4):
                nxt = self.move_once(state, d)
                if nxt in parent:
                    continue

                parent[nxt] = (state, d)
                nd = dep + 1

                if self.is_goal_state(nxt):
                    return self.reconstruct_bfs_path(parent, nxt)

                q.append((nxt, nd))
                visited_nodes += 1
                if visited_nodes >= node_limit:
                    return None

        return None

    def reduce_once(self, start_mask: int, seed: int):
        rng = random.Random(seed)

        current = start_mask
        answer = []

        MAX_LEN = 150
        REDUCE_LIMIT = 60
        TARGET = 12

        # Etap 1: kilka mocnych docisków
        long_candidates = self.long_pushes[:]
        rng.shuffle(long_candidates)

        for seq in long_candidates[:8]:
            if len(answer) + len(seq) > REDUCE_LIMIT:
                continue
            nxt = self.apply_seq(current, seq)
            if self.score_mask(nxt) < self.score_mask(current):
                current = nxt
                answer.extend(DIR_CHARS[d] for d in seq)
                if self.is_goal_state(current):
                    return "".join(answer), current

        # Etap 2: lokalna redukcja
        stagnation = 0
        while len(answer) < REDUCE_LIMIT and popcount(current) > TARGET:
            best_seq = None
            best_mask = current
            best_score = self.score_mask(current)

            # losujemy podzbiór krótkich sekwencji zamiast sprawdzać wszystkie
            sample = rng.sample(self.short_seqs, min(24, len(self.short_seqs)))

            # kilka prostych ruchów zawsze warto sprawdzić
            core = [(0,), (1,), (2,), (3,)]
            checked = core + sample

            for seq in checked:
                if len(answer) + len(seq) > REDUCE_LIMIT:
                    continue
                nxt = self.apply_seq(current, seq)
                sc = self.score_mask(nxt)
                if sc < best_score:
                    best_score = sc
                    best_seq = seq
                    best_mask = nxt

            if best_seq is None:
                stagnation += 1
                if stagnation >= 3:
                    break

                # ruch perturbacyjny
                seq = rng.choice(self.long_pushes)
                if len(answer) + len(seq) > REDUCE_LIMIT:
                    break
                current = self.apply_seq(current, seq)
                answer.extend(DIR_CHARS[d] for d in seq)
            else:
                stagnation = 0
                current = best_mask
                answer.extend(DIR_CHARS[d] for d in best_seq)

            if self.is_goal_state(current):
                return "".join(answer), current

        return "".join(answer), current

    def solve(self):
        if self.is_goal_state(self.start_mask):
            return ""

        best_prefix = ""
        best_mask = self.start_mask
        best_score = self.score_mask(self.start_mask)

        # szybka próba BFS od razu, jeśli stan mały
        if popcount(self.start_mask) <= 18:
            path = self.bounded_bfs(self.start_mask, 150, 300000)
            if path is not None:
                return path

        # kilka restartów redukcji
        for seed in range(12):
            prefix, mask = self.reduce_once(self.start_mask, seed)
            sc = self.score_mask(mask)
            if sc < best_score or (sc == best_score and len(prefix) < len(best_prefix)):
                best_score = sc
                best_prefix = prefix
                best_mask = mask

            remaining = 150 - len(prefix)
            if remaining < 0:
                continue

            cnt = popcount(mask)
            if cnt <= 10:
                limit = 1000000
            elif cnt <= 14:
                limit = 700000
            elif cnt <= 18:
                limit = 400000
            elif cnt <= 24:
                limit = 200000
            else:
                continue

            path = self.bounded_bfs(mask, remaining, limit)
            if path is not None and len(prefix) + len(path) <= 150:
                return prefix + path

        # ostatnia próba z najlepszego znalezionego stanu
        remaining = 150 - len(best_prefix)
        if remaining >= 0:
            path = self.bounded_bfs(best_mask, remaining, 1200000)
            if path is not None and len(best_prefix) + len(path) <= 150:
                return best_prefix + path

        return best_prefix[:150]


def main():
    lines = read_input()
    solver = Solver(lines)
    ans = solver.solve()
    write_output(ans)


if __name__ == "__main__":
    main()