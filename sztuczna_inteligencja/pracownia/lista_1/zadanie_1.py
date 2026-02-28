# Zadanie 1 (K+W vs K, kooperacyjne) - rozwiązanie w Pythonie.
#
# Pomysł:
# - Stan gry opisujemy jako (tura, WK, WR, BK), gdzie:
#   - tura: 'white' albo 'black'
#   - WK: pozycja białego króla (0..63)
#   - WR: pozycja białej wieży (0..63) lub None (gdy wieża została zbita)
#   - BK: pozycja czarnego króla (0..63)
# - Ponieważ to mat pomocniczy (czarny współpracuje), obie strony wybierają ruchy
#   prowadzące jak najszybciej do mata, więc szukamy najkrótszej ścieżki w grafie stanów.
# - Używamy BFS od stanu startowego po wszystkich legalnych ruchach.
# - Stan wygrywający (cel BFS): czarne na ruchu i są matowane, tzn.:
#   - BK jest w szachu (od WR lub WK)
#   - czarne nie mają żadnego legalnego ruchu
# - Stan końcowy niewygrywający: pat (strona na ruchu nie ma ruchu i nie jest w szachu).
#   Takie stany traktujemy jako "ślepe zaułki" (nie cel).
# - Jeśli BFS nie znajdzie mata, wypisujemy "INF".
#
# Dodatkowo:
# - Tryb wsadowy: czyta zad1_input.txt, pisze zad1_output.txt.
# - Tryb debug: uruchom z argumentem "--debug", wtedy:
#   - czyta pierwszą pozycję z zad1_input.txt
#   - wypisuje kolejne stany prowadzące do mata (jeśli istnieje).
#
# Ważne zasady legalności (bez bibliotek szachowych):
# - Króle nie mogą stać obok siebie (na pola sąsiadujące).
# - Król nie może wejść na pole szachowane przez przeciwnika.
# - Wieża porusza się po linii prostej i nie przeskakuje figur.
# - Czarny król może zbić wieżę, jeśli po zbiciu nie stoi w szachu białego króla.

from collections import deque
import sys

FILES = "abcdefgh"


# ----------------------------
# Konwersje pól (np. "c4") <-> indeks (0..63)
# ----------------------------

def sq_to_idx(sq: str) -> int:
    """Zamienia np. 'a1' na indeks 0..63."""
    file_char = sq[0]
    rank_char = sq[1]
    x = FILES.index(file_char)          # 0..7
    y = int(rank_char) - 1              # 0..7
    return y * 8 + x


def idx_to_sq(i: int) -> str:
    """Zamienia indeks 0..63 na np. 'a1'."""
    x = i % 8
    y = i // 8
    return FILES[x] + str(y + 1)


def idx_xy(i: int) -> tuple[int, int]:
    """Indeks 0..63 -> (x,y)."""
    return (i % 8, i // 8)


def xy_idx(x: int, y: int) -> int:
    """(x,y) -> indeks 0..63."""
    return y * 8 + x


def in_board(x: int, y: int) -> bool:
    return 0 <= x < 8 and 0 <= y < 8


# ----------------------------
# Ataki (szachowanie)
# ----------------------------

KING_DIRS = [(-1, -1), (0, -1), (1, -1),
             (-1,  0),          (1,  0),
             (-1,  1), (0,  1), (1,  1)]


def kings_adjacent(wk: int, bk: int) -> bool:
    """Czy króle stoją na sąsiadujących polach?"""
    wx, wy = idx_xy(wk)
    bx, by = idx_xy(bk)
    return max(abs(wx - bx), abs(wy - by)) <= 1


def rook_attacks_square(wr: int | None, wk: int, bk: int, target: int) -> bool:
    """
    Czy biała wieża atakuje pole target?
    Uwaga: atak jest blokowany przez figury (WK i BK).
    """
    if wr is None:
        return False

    rx, ry = idx_xy(wr)
    tx, ty = idx_xy(target)

    # Musi być ta sama kolumna albo wiersz.
    if rx != tx and ry != ty:
        return False

    # Idziemy od WR w stronę target i sprawdzamy czy coś stoi po drodze.
    dx = 0 if rx == tx else (1 if tx > rx else -1)
    dy = 0 if ry == ty else (1 if ty > ry else -1)

    x, y = rx + dx, ry + dy
    while (x, y) != (tx, ty):
        sq = xy_idx(x, y)
        if sq == wk or sq == bk:
            return False
        x += dx
        y += dy

    return True


def white_attacks_square(wk: int, wr: int | None, bk: int, target: int) -> bool:
    """Czy białe atakują pole target? (WK lub WR)."""
    # Atak króla: sąsiedztwo
    wx, wy = idx_xy(wk)
    tx, ty = idx_xy(target)
    if max(abs(wx - tx), abs(wy - ty)) == 1:
        return True

    # Atak wieży: linia prosta bez blokady
    return rook_attacks_square(wr, wk, bk, target)


def black_attacks_square(bk: int, target: int) -> bool:
    """Czarny ma tylko króla - atakuje pola sąsiednie."""
    bx, by = idx_xy(bk)
    tx, ty = idx_xy(target)
    return max(abs(bx - tx), abs(by - ty)) == 1


def black_in_check(wk: int, wr: int | None, bk: int) -> bool:
    """Czy BK jest w szachu?"""
    return white_attacks_square(wk, wr, bk, bk)


# ----------------------------
# Generowanie legalnych ruchów
# ----------------------------

def gen_white_king_moves(wk: int, wr: int | None, bk: int):
    """Generuje legalne ruchy białego króla."""
    wx, wy = idx_xy(wk)
    for dx, dy in KING_DIRS:
        nx, ny = wx + dx, wy + dy
        if not in_board(nx, ny):
            continue
        n = xy_idx(nx, ny)

        # Nie wchodzimy na zajęte pole (WR albo BK).
        if n == bk or (wr is not None and n == wr):
            continue

        # Króle nie mogą być sąsiadami.
        if max(abs(nx - idx_xy(bk)[0]), abs(ny - idx_xy(bk)[1])) <= 1:
            continue

        # Dodatkowy warunek: biały król nie może wejść na pole atakowane przez BK,
        # ale to i tak jest równoważne warunkowi "króle nie sąsiadują".
        yield n


def gen_white_rook_moves(wk: int, wr: int, bk: int):
    """Generuje legalne ruchy białej wieży (bez bicia króla)."""
    rx, ry = idx_xy(wr)

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        x, y = rx + dx, ry + dy
        while in_board(x, y):
            n = xy_idx(x, y)

            # Nie można wejść na WK.
            if n == wk:
                break

            # Nie można wejść na BK (wieża nie bije króla); BK blokuje linię.
            if n == bk:
                break

            # Pole puste -> legalny ruch.
            yield n

            x += dx
            y += dy


def gen_black_king_moves(wk: int, wr: int | None, bk: int):
    """Generuje legalne ruchy czarnego króla (w tym bicie wieży)."""
    bx, by = idx_xy(bk)
    for dx, dy in KING_DIRS:
        nx, ny = bx + dx, by + dy
        if not in_board(nx, ny):
            continue
        n = xy_idx(nx, ny)

        # BK nie może wejść na WK.
        if n == wk:
            continue

        # Króle nie mogą być sąsiadami.
        wx, wy = idx_xy(wk)
        if max(abs(nx - wx), abs(ny - wy)) == 1:
            continue

        # Jeśli BK wchodzi na WR, to jest bicie wieży (WR -> None).
        next_wr = wr
        if wr is not None and n == wr:
            next_wr = None

        # BK nie może wejść na pole szachowane przez białe.
        if white_attacks_square(wk, next_wr, n, n):
            continue

        yield n, next_wr


def legal_moves(state):
    """
    Zwraca listę następnych stanów osiągalnych jednym legalnym ruchem.
    Stan: (turn, wk, wr, bk)
    """
    turn, wk, wr, bk = state
    res = []

    if turn == "white":
        # Ruch białego króla
        for nwk in gen_white_king_moves(wk, wr, bk):
            res.append(("black", nwk, wr, bk))

        # Ruch białej wieży (jeśli żyje)
        if wr is not None:
            for nwr in gen_white_rook_moves(wk, wr, bk):
                res.append(("black", wk, nwr, bk))

    else:
        # Ruch czarnego króla (z ewentualnym biciem WR)
        for nbk, nwr in gen_black_king_moves(wk, wr, bk):
            res.append(("white", wk, nwr, nbk))

    return res


# ----------------------------
# Mat, pat
# ----------------------------

def is_checkmate(state) -> bool:
    """Czy w tym stanie czarne są zamatowane? (czarne na ruchu, szach i brak ruchów)."""
    turn, wk, wr, bk = state
    if turn != "black":
        return False
    if not black_in_check(wk, wr, bk):
        return False
    # Jeśli czarne nie mają legalnego ruchu - mat.
    return len(legal_moves(state)) == 0


def is_stalemate(state) -> bool:
    """Pat: strona na ruchu nie ma legalnego ruchu i nie jest w szachu."""
    turn, wk, wr, bk = state
    if len(legal_moves(state)) != 0:
        return False

    if turn == "black":
        return not black_in_check(wk, wr, bk)
    else:
        # Biały król byłby w szachu tylko gdyby króle stały obok siebie,
        # co i tak jest pozycją nielegalną. Zostawiamy definicję konsekwentnie:
        return not black_attacks_square(bk, wk)


# ----------------------------
# BFS (najkrótsza liczba półruchów do mata)
# ----------------------------

def bfs_min_to_mate(start_state, want_path: bool = False):
    """
    BFS od startu do dowolnego checkmate.
    Zwraca:
    - (dist, path_states) jeśli want_path=True i istnieje rozwiązanie
    - (dist, None) jeśli want_path=False i istnieje rozwiązanie
    - ("INF", None) jeśli brak rozwiązania
    """
    if is_checkmate(start_state):
        return 0, [start_state] if want_path else None

    q = deque([start_state])
    dist = {start_state: 0}
    parent = {start_state: None} if want_path else None

    while q:
        cur = q.popleft()
        d = dist[cur]

        # Jeżeli weszliśmy w pat - nie rozwijamy dalej tego stanu.
        if is_stalemate(cur):
            continue

        for nxt in legal_moves(cur):
            if nxt in dist:
                continue
            dist[nxt] = d + 1
            if want_path:
                parent[nxt] = cur

            if is_checkmate(nxt):
                if not want_path:
                    return dist[nxt], None

                # Rekonstrukcja ścieżki
                path = []
                x = nxt
                while x is not None:
                    path.append(x)
                    x = parent[x]
                path.reverse()
                return dist[nxt], path

            q.append(nxt)

    return "INF", None


# ----------------------------
# Debug: proste "rysowanie" planszy
# ----------------------------

def print_board(state):
    """Wypisuje planszę 8x8 jako tekst."""
    turn, wk, wr, bk = state
    board = [["." for _ in range(8)] for _ in range(8)]

    wx, wy = idx_xy(wk)
    board[wy][wx] = "♔"   # white King

    if wr is not None:
        rx, ry = idx_xy(wr)
        board[ry][rx] = "♖"   # white Rook

    bx, by = idx_xy(bk)
    board[by][bx] = "♚"   # black King

    print(f"Turn: {turn}")
    for y in range(7, -1, -1):
        print(str(y + 1) + " " + " ".join(board[y]))
    print("  " + " ".join(list(FILES)))
    print()


# ----------------------------
# IO: wsad + debug
# ----------------------------

def parse_line(line: str):
    """
    Parsuje linię: "<turn> <WK> <WR> <BK>"
    np. "black c4 c8 h3"
    """
    parts = line.strip().split()
    if not parts:
        return None
    turn = parts[0]
    wk = sq_to_idx(parts[1])
    wr = sq_to_idx(parts[2])
    bk = sq_to_idx(parts[3])

    # Podstawowa walidacja pozycji:
    # - figury na różnych polach
    # - króle nie sąsiadują
    if wk == bk or wk == wr or bk == wr:
        # pozycja nielegalna -> nie da się sensownie liczyć, traktujemy jako INF
        return ("ILLEGAL",)
    if kings_adjacent(wk, bk):
        return ("ILLEGAL",)

    return (turn, wk, wr, bk)


def batch_mode(in_path="zad1_input.txt", out_path="zad1_output.txt"):
    out_lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            if parsed[0] == "ILLEGAL":
                out_lines.append("INF")
                continue

            ans, _ = bfs_min_to_mate(parsed, want_path=False)
            out_lines.append(str(ans))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


def debug_mode(in_path="zad1_input.txt"):
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            if parsed[0] == "ILLEGAL":
                print("Pozycja nielegalna -> INF")
                return

            ans, path = bfs_min_to_mate(parsed, want_path=True)
            print("Wynik:", ans)
            if ans == "INF":
                return

            for i, st in enumerate(path):
                print(f"Krok {i} (po {i} półruchach):")
                print_board(st)
            return


def main():
    # Jeśli podano "--debug", robimy tryb debug.
    if len(sys.argv) >= 2 and sys.argv[1] == "--debug":
        debug_mode()
    else:
        batch_mode()


if __name__ == "__main__":
    main()