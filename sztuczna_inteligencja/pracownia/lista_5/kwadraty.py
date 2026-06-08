import random
import math
import time
import sys

def main():
    N = 24
    B = 70

    board = [[-1] * B for _ in range(B)]  # -1 = puste, 0..23 = indeks kwadratu
    pos = [None] * N                        # pos[i] = (x, y) lewy górny róg
    empty = [B * B]                         # licznik pustych pól (mutowalny)

    def can_place(x, y, s):
        if x + s > B or y + s > B:
            return False
        for r in range(y, y + s):
            row = board[r]
            for c in range(x, x + s):
                if row[c] != -1:
                    return False
        return True

    def place(i, x, y):
        s = i + 1
        for r in range(y, y + s):
            row = board[r]
            for c in range(x, x + s):
                if row[c] == -1:
                    empty[0] -= 1
                row[c] = i
        pos[i] = (x, y)

    def unplace(i):
        x, y = pos[i]
        s = i + 1
        for r in range(y, y + s):
            row = board[r]
            for c in range(x, x + s):
                row[c] = -1
                empty[0] += 1
        pos[i] = None

    # Algorytm 1: Zachłanny — od największego do najmniejszego, pozycja bottom-left
    for i in sorted(range(N), key=lambda k: -k):
        s = i + 1
        for y in range(B - s + 1):
            for x in range(B - s + 1):
                if can_place(x, y, s):
                    place(i, x, y)
                    break
            else:
                continue
            break

    best = empty[0]
    best_board = [row[:] for row in board]
    print(f"# Po greedy: {best} wolnych pól", file=sys.stderr)

    # Algorytm 2: Symulowane wyżarzanie
    T_start = 500.0
    t0 = time.time()
    limit = 28.0
    iters = 0

    while time.time() - t0 < limit:
        iters += 1
        progress = (time.time() - t0) / limit
        T = T_start * (0.001 ** progress)  # chłodzenie geometryczne

        i = random.randrange(N)
        if pos[i] is None:
            continue

        ox, oy = pos[i]
        s = i + 1
        before = empty[0]

        unplace(i)

        # Losowy ruch: albo całkowicie losowy, albo małe przesunięcie
        if random.random() < 0.5:
            nx = random.randrange(B - s + 1)
            ny = random.randrange(B - s + 1)
        else:
            nx = max(0, min(B - s, ox + random.randint(-8, 8)))
            ny = max(0, min(B - s, oy + random.randint(-8, 8)))

        if can_place(nx, ny, s):
            place(i, nx, ny)
            delta = empty[0] - before
            # Kryterium akceptacji Metropolisa
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                if empty[0] < best:
                    best = empty[0]
                    best_board = [row[:] for row in board]
                    print(f"# iter={iters} t={time.time()-t0:.1f}s  wolne={best}", file=sys.stderr)
                    if best == 0:
                        break
            else:
                unplace(i)
                place(i, ox, oy)
        else:
            place(i, ox, oy)

    print(f"# Łącznie: {iters} iteracji, {time.time()-t0:.1f}s", file=sys.stderr)

    # Wypisz wynik zgodnie ze specyfikacją
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWX'
    print(best)
    for row in best_board:
        print(''.join(chars[c] if c >= 0 else '.' for c in row))

main()