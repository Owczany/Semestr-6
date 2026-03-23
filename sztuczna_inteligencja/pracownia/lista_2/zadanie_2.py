from collections import deque
from itertools import product, permutations
import os
import sys


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_CHARS = "UDLR"
DIR_INDEX = {c: i for i, c in enumerate(DIR_CHARS)}


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
    if os.path.exists("zad_input.txt") or not sys.stdout.isatty():
        try:
            with open("zad_output.txt", "w", encoding="utf-8") as f:
                f.write(ans + "\n")
            return
        except Exception:
            pass
    sys.stdout.write(ans + "\n")


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
                ch = self.grid[r][c]
                if ch != '#':
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

        self.short_sequences = self._build_short_sequences()
        self.macro_sequences = self._build_macro_sequences()

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

    def _build_short_sequences(self):
        seqs = []
        for length in (1, 2, 3):
            for tup in product(range(4), repeat=length):
                seqs.append(tup)
        return seqs

    def _build_macro_sequences(self):
        reps = {
            0: self.h,  # U
            1: self.h,  # D
            2: self.w,  # L
            3: self.w,  # R
        }

        seqs = []

        for d in range(4):
            seqs.append(tuple([d] * reps[d]))

        for perm in permutations(range(4)):
            seq = []
            for d in perm:
                seq.extend([d] * reps[d])
            seqs.append(tuple(seq))

        seen = set()
        uniq = []
        for s in seqs:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

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
        s = 0
        mx = 0
        for i in iter_bits(mask):
            d = self.dist_to_goal[i]
            s += d
            if d > mx:
                mx = d
        return (cnt, s, mx)

    def bounded_bfs(self, start_mask: int, max_depth: int, node_limit: int):
        if self.is_goal_state(start_mask):
            return ""

        q = deque()
        q.append(start_mask)

        parent = {start_mask: (-1, -1)}
        depth = {start_mask: 0}

        visited_nodes = 1

        while q:
            state = q.popleft()
            dep = depth[state]
            if dep >= max_depth:
                continue

            for d in range(4):
                nxt = self.move_once(state, d)
                if nxt in parent:
                    continue
                parent[nxt] = (state, d)
                nd = dep + 1
                depth[nxt] = nd

                if self.is_goal_state(nxt):
                    path = []
                    cur = nxt
                    while parent[cur][0] != -1:
                        prev, move = parent[cur]
                        path.append(DIR_CHARS[move])
                        cur = prev
                    path.reverse()
                    return "".join(path)

                q.append(nxt)
                visited_nodes += 1
                if visited_nodes >= node_limit:
                    return None

        return None

    def choose_best_reduction(self, mask: int, remaining_budget: int):
        best_seq = None
        best_mask = None
        best_score = self.score_mask(mask)

        # Najpierw mocne makroruchy
        for seq in self.macro_sequences:
            if len(seq) > remaining_budget:
                continue
            nxt = self.apply_seq(mask, seq)
            sc = self.score_mask(nxt)
            if sc < best_score:
                best_score = sc
                best_seq = seq
                best_mask = nxt

        # Potem lokalne zachłanne sekwencje
        for seq in self.short_sequences:
            if len(seq) > remaining_budget:
                continue
            nxt = self.apply_seq(mask, seq)
            sc = self.score_mask(nxt)
            if sc < best_score:
                best_score = sc
                best_seq = seq
                best_mask = nxt

        return best_seq, best_mask, best_score

    def solve(self):
        if self.is_goal_state(self.start_mask):
            return ""

        current = self.start_mask
        answer = []

        MAX_LEN = 150
        PHASE1_LIMIT = 130
        BFS_THRESHOLD = 2 # Z testów dla 3 działa

        # Spróbuj od razu BFS, jeśli stan jest mały
        if popcount(current) <= BFS_THRESHOLD:
            path = self.bounded_bfs(current, MAX_LEN, 250000)
            if path is not None and len(path) <= MAX_LEN:
                return path

        # Faza 1 - redukcja niepewności
        improved = True
        while improved and len(answer) < PHASE1_LIMIT:
            improved = False

            remaining_total = MAX_LEN - len(answer)

            # Jeśli stan już mały, próbujemy BFS
            if popcount(current) <= BFS_THRESHOLD:
                path = self.bounded_bfs(current, remaining_total, 300000)
                if path is not None and len(answer) + len(path) <= MAX_LEN:
                    return "".join(answer) + path

            # Szukamy najlepszej redukcji
            seq, nxt, sc = self.choose_best_reduction(
                current,
                min(remaining_total, PHASE1_LIMIT - len(answer))
            )

            if seq is not None and nxt != current:
                for d in seq:
                    answer.append(DIR_CHARS[d])
                current = nxt
                improved = True

                if self.is_goal_state(current):
                    return "".join(answer)

        # Po redukcji próbujemy BFS kilkoma progami
        remaining_total = MAX_LEN - len(answer)

        for limit, threshold in [
            (120000, 20),
            (250000, 24),
            (500000, 28),
            (800000, 40),
        ]:
            if popcount(current) <= threshold:
                path = self.bounded_bfs(current, remaining_total, limit)
                if path is not None and len(answer) + len(path) <= MAX_LEN:
                    return "".join(answer) + path

        # Awaryjnie: jeszcze kilka lokalnych ruchów zachłannych,
        # jednocześnie co chwilę próbując BFS
        # while len(answer) < MAX_LEN:
        #     remaining_total = MAX_LEN - len(answer)

        #     path = self.bounded_bfs(current, remaining_total, 300000)
        #     if path is not None and len(answer) + len(path) <= MAX_LEN:
        #         return "".join(answer) + path

        #     best_seq = None
        #     best_mask = None
        #     best_score = self.score_mask(current)

        #     for seq in self.short_sequences:
        #         if len(seq) > remaining_total:
        #             continue
        #         nxt = self.apply_seq(current, seq)
        #         sc = self.score_mask(nxt)
        #         if sc < best_score:
        #             best_score = sc
        #             best_seq = seq
        #             best_mask = nxt

        #     if best_seq is None or best_mask == current:
        #         break

        #     for d in best_seq:
        #         answer.append(DIR_CHARS[d])
        #     current = best_mask

        #     if self.is_goal_state(current):
        #         return "".join(answer)

        # # Ostatnia próba BFS
        # remaining_total = MAX_LEN - len(answer)
        # path = self.bounded_bfs(current, remaining_total, 1000000)
        # if path is not None and len(answer) + len(path) <= MAX_LEN:
        #     return "".join(answer) + path

        # # W praktyce dla sensownych testów nie powinno tu dojść.
        # # Gdyby jednak doszło, zwracamy to, co mamy.
        # return "".join(answer[:MAX_LEN])


def main():
    lines = read_input()
    solver = Solver(lines)
    ans = solver.solve()
    write_output(ans)


if __name__ == "__main__":
    main()