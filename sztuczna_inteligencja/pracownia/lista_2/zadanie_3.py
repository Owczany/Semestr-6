from collections import deque
import heapq
import os
import sys


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_CHARS = "UDLR"


def iter_bits(mask: int):
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


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
    try:
        with open("zad_output.txt", "w", encoding="utf-8") as f:
            f.write(ans + "\n")
    except Exception:
        sys.stdout.write(ans + "\n")


class Solver:
    def __init__(self, lines):
        self.grid = lines
        self.h = len(lines)
        self.w = len(lines[0]) if self.h else 0

        self.pos_to_id = {}
        self.id_to_pos = []

        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r][c] != '#':
                    idx = len(self.id_to_pos)
                    self.pos_to_id[(r, c)] = idx
                    self.id_to_pos.append((r, c))

        self.n = len(self.id_to_pos)

        self.start_mask = 0
        self.goal_mask = 0

        for idx, (r, c) in enumerate(self.id_to_pos):
            ch = self.grid[r][c]
            if ch in ('S', 'B'):
                self.start_mask |= 1 << idx
            if ch in ('G', 'B'):
                self.goal_mask |= 1 << idx

        self.next_pos = [[0] * self.n for _ in range(4)]
        for idx, (r, c) in enumerate(self.id_to_pos):
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.pos_to_id:
                    self.next_pos[d][idx] = self.pos_to_id[(nr, nc)]
                else:
                    self.next_pos[d][idx] = idx

        self.dist_to_goal = self._compute_dist_to_goal()
        self.move_cache = [dict() for _ in range(4)]
        self.h_cache = {}

    def _compute_dist_to_goal(self):
        INF = 10**9
        dist = [INF] * self.n
        q = deque()

        for i in range(self.n):
            if (self.goal_mask >> i) & 1:
                dist[i] = 0
                q.append(i)

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

    def is_goal_state(self, mask: int) -> bool:
        return (mask & ~self.goal_mask) == 0

    def move_once(self, mask: int, d: int) -> int:
        cached = self.move_cache[d].get(mask)
        if cached is not None:
            return cached

        nxt = 0
        trans = self.next_pos[d]
        for i in iter_bits(mask):
            nxt |= 1 << trans[i]

        self.move_cache[d][mask] = nxt
        return nxt

    def heuristic(self, mask: int) -> int:
        cached = self.h_cache.get(mask)
        if cached is not None:
            return cached

        h = 0
        for i in iter_bits(mask):
            if self.dist_to_goal[i] > h:
                h = self.dist_to_goal[i]

        self.h_cache[mask] = h
        return h

    def reconstruct_path(self, parent, end_state):
        path = []
        cur = end_state
        while parent[cur][0] != -1:
            prev, move = parent[cur]
            path.append(DIR_CHARS[move])
            cur = prev
        path.reverse()
        return "".join(path)

    def astar(self):
        start = self.start_mask

        if self.is_goal_state(start):
            return ""

        pq = []
        g_score = {start: 0}
        parent = {start: (-1, -1)}
        counter = 0

        h0 = self.heuristic(start)
        heapq.heappush(pq, (h0, 0, counter, start))

        while pq:
            f, g, _, state = heapq.heappop(pq)

            if g != g_score.get(state):
                continue

            if self.is_goal_state(state):
                return self.reconstruct_path(parent, state)

            for d in range(4):
                nxt = self.move_once(state, d)
                ng = g + 1

                old = g_score.get(nxt)
                if old is None or ng < old:
                    g_score[nxt] = ng
                    parent[nxt] = (state, d)
                    counter += 1
                    h = self.heuristic(nxt)
                    heapq.heappush(pq, (ng + h, ng, counter, nxt))

        return ""

    def solve(self):
        return self.astar()


def main():
    lines = read_input()
    solver = Solver(lines)
    ans = solver.solve()
    write_output(ans)


if __name__ == "__main__":
    main()