# Zadanie 5 (4p) - uproszczone nonogramy (w każdym wierszu/kolumnie co najwyżej 1 blok).
#
# Wariant zadania:
# - Plansza ma rozmiar X (liczba wierszy) oraz Y (liczba kolumn).
# - Dla każdego wiersza i każdej kolumny dana jest jedna liczba D:
#   * D = 0 oznacza brak czarnych pól w tej linii (same zera)
#   * D > 0 oznacza dokładnie jeden spójny blok jedynek długości D (reszta zera)
#
# Algorytm (lokalne przeszukiwanie, inspirowane WalkSat):
# 1) Startujemy od pewnego wypełnienia planszy (losowo albo same zera).
# 2) Iteracyjnie:
#    a) losujemy wiersz lub kolumnę, która jest NIEZGODNA (ma koszt > 0)
#    b) w tej linii wybieramy pole (i,j), którego flip (0<->1) najbardziej poprawia
#       sumaryczny koszt: koszt_wiersza_i + koszt_kolumny_j
#    c) z małym prawdopodobieństwem robimy krok "gorszy" (szum) - np. losowy piksel
#       albo optymalizujemy tylko wiersz (ignorujemy kolumnę) itd.
# 3) Jeśli przez dłuższy czas nie ma poprawy - restart (losujemy planszę od nowa).
#
# Kluczowa funkcja kosztu dla linii:
# - opt_dist(bits, D) = minimalna liczba flipów w tej linii, żeby spełniała warunek bloku D.
#   (To jest dokładnie funkcja z Zadania 4.)
#
# Wejście:  zad5_input.txt
# Wyjście:  zad5_output.txt
#
# Uwaga praktyczna:
# - Sprawdzarka ocenia ile testów rozwiążesz w limicie czasu, więc lepiej:
#   * szybko znajdować rozwiązanie dla małych/średnich plansz
#   * mieć restart i trochę losowości, żeby wyjść z lokalnych minimów

import os
import random
from typing import List, Tuple


# ----------------------------
# Koszt linii: opt_dist (z Zad.4)
# ----------------------------

def opt_dist(bits: List[int], D: int) -> int:
    """
    Minimalna liczba flipów, aby uzyskać:
    - D>0: jeden spójny blok 1 długości D, poza blokiem same 0
    - D=0: same 0
    """
    n = len(bits)
    if D == 0:
        return sum(bits)
    if D > n:
        # Sprzeczna specyfikacja - nie powinna wystąpić w testach.
        return 10**9

    # pref[i] = liczba jedynek w bits[0:i]
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + bits[i]
    ones_total = pref[n]

    best = 10**18
    for start in range(0, n - D + 1):
        end = start + D
        ones_in = pref[end] - pref[start]
        # cost = zeros_in_window + ones_outside
        cost = (D - ones_in) + (ones_total - ones_in)
        if cost < best:
            best = cost

    return int(best)


# ----------------------------
# I/O
# ----------------------------

def read_instance(path: str) -> Tuple[int, int, List[int], List[int]]:
    """
    Czyta instancję:
      X Y
      (X linii opisów wierszy)
      (Y linii opisów kolumn)
    Zwraca: X, Y, row_desc[X], col_desc[Y]
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    X, Y = map(int, lines[0].split())
    row_desc = [int(lines[1 + i]) for i in range(X)]
    col_desc = [int(lines[1 + X + j]) for j in range(Y)]
    return X, Y, row_desc, col_desc


def write_solution(path: str, grid: List[List[int]]) -> None:
    """
    Zapisuje rozwiązanie:
    - X linii (wiersze)
    - w każdej linii Y znaków:
      '.' dla 0, '#' dla 1
    """
    X = len(grid)
    Y = len(grid[0]) if X else 0
    with open(path, "w", encoding="utf-8") as f:
        for i in range(X):
            row = "".join("#" if grid[i][j] == 1 else "." for j in range(Y))
            f.write(row + "\n")


# ----------------------------
# Funkcje pomocnicze: koszty i sprawdzanie
# ----------------------------

def compute_row_costs(grid: List[List[int]], row_desc: List[int]) -> List[int]:
    """Liczy koszt każdego wiersza niezależnie."""
    X = len(grid)
    return [opt_dist(grid[i], row_desc[i]) for i in range(X)]


def compute_col_costs(grid: List[List[int]], col_desc: List[int]) -> List[int]:
    """Liczy koszt każdej kolumny niezależnie."""
    X = len(grid)
    Y = len(grid[0])
    costs = []
    for j in range(Y):
        col_bits = [grid[i][j] for i in range(X)]
        costs.append(opt_dist(col_bits, col_desc[j]))
    return costs


def total_cost(row_costs: List[int], col_costs: List[int]) -> int:
    """Sumaryczny koszt dopasowania planszy do specyfikacji."""
    return sum(row_costs) + sum(col_costs)


def random_initial_grid(X: int, Y: int, rng: random.Random, p: float = 0.5) -> List[List[int]]:
    """Losowe wypełnienie planszy; p = prawdopodobieństwo jedynki."""
    return [[1 if rng.random() < p else 0 for _ in range(Y)] for _ in range(X)]


# ----------------------------
# Lokalna poprawa: wybór ruchu (flip jednego pola)
# ----------------------------

def best_flip_in_row(
    grid: List[List[int]],
    i: int,
    row_desc: List[int],
    col_desc: List[int],
    row_costs: List[int],
    col_costs: List[int],
    rng: random.Random,
    noise_random_pixel: float,
    noise_row_only: float
) -> Tuple[int, int]:
    """
    Wybiera najlepszy flip w wierszu i.

    Zgodnie z algorytmem:
    - normalnie: wybieramy j, które minimalizuje (koszt_wiersza_i + koszt_kolumny_j) po flipie
    - z małym prawdopodobieństwem:
      * noise_random_pixel: wybieramy losowe j (nieoptymalnie)
      * noise_row_only: wybieramy j optymalizujące tylko wiersz (ignorujemy kolumnę)
    """
    Y = len(grid[0])

    # 1) Szum: losowy piksel
    if rng.random() < noise_random_pixel:
        return i, rng.randrange(Y)

    # 2) Normalnie sprawdzamy wszystkie j i wybieramy najlepsze
    best_js = []
    best_delta = None

    # Czy ignorujemy kolumnę?
    row_only = (rng.random() < noise_row_only)

    for j in range(Y):
        old_val = grid[i][j]
        new_val = 1 - old_val

        # Koszt wiersza po flipie:
        tmp_row = grid[i][:]
        tmp_row[j] = new_val
        new_row_cost = opt_dist(tmp_row, row_desc[i])

        # Koszt kolumny po flipie:
        if row_only:
            new_col_cost = col_costs[j]
        else:
            X = len(grid)
            col_bits = [grid[r][j] for r in range(X)]
            col_bits[i] = new_val
            new_col_cost = opt_dist(col_bits, col_desc[j])

        # Delta = nowy - stary (chcemy jak najbardziej ujemne)
        delta = (new_row_cost - row_costs[i]) + (new_col_cost - col_costs[j])

        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_js = [j]
        elif delta == best_delta:
            best_js.append(j)

    # Jeśli jest kilka równie dobrych - wybieramy losowo (pomaga uniknąć cykli)
    j = rng.choice(best_js)
    return i, j


def best_flip_in_col(
    grid: List[List[int]],
    j: int,
    row_desc: List[int],
    col_desc: List[int],
    row_costs: List[int],
    col_costs: List[int],
    rng: random.Random,
    noise_random_pixel: float,
    noise_col_only: float
) -> Tuple[int, int]:
    """
    Symetrycznie: wybiera najlepszy flip w kolumnie j.
    """
    X = len(grid)

    if rng.random() < noise_random_pixel:
        return rng.randrange(X), j

    best_is = []
    best_delta = None
    col_only = (rng.random() < noise_col_only)

    for i in range(X):
        old_val = grid[i][j]
        new_val = 1 - old_val

        # Kolumna po flipie
        col_bits = [grid[r][j] for r in range(X)]
        col_bits[i] = new_val
        new_col_cost = opt_dist(col_bits, col_desc[j])

        # Wiersz po flipie
        if col_only:
            new_row_cost = row_costs[i]
        else:
            tmp_row = grid[i][:]
            tmp_row[j] = new_val
            new_row_cost = opt_dist(tmp_row, row_desc[i])

        delta = (new_row_cost - row_costs[i]) + (new_col_cost - col_costs[j])

        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_is = [i]
        elif delta == best_delta:
            best_is.append(i)

    i = rng.choice(best_is)
    return i, j


def apply_flip_and_update_costs(
    grid: List[List[int]],
    i: int,
    j: int,
    row_desc: List[int],
    col_desc: List[int],
    row_costs: List[int],
    col_costs: List[int],
) -> None:
    """
    Wykonuje flip (i,j) i aktualizuje tylko koszt wiersza i oraz kolumny j,
    bo reszta linii nie zmienia się.
    """
    X = len(grid)
    old_val = grid[i][j]
    grid[i][j] = 1 - old_val

    # Aktualizacja kosztu wiersza i
    row_costs[i] = opt_dist(grid[i], row_desc[i])

    # Aktualizacja kosztu kolumny j
    col_bits = [grid[r][j] for r in range(X)]
    col_costs[j] = opt_dist(col_bits, col_desc[j])


# ----------------------------
# Główne rozwiązywanie (WalkSat-like)
# ----------------------------

def solve_nonogram(
    X: int,
    Y: int,
    row_desc: List[int],
    col_desc: List[int],
    time_budget_steps: int = 200000,
    restart_after_no_improve: int = 5000,
    noise_random_pixel: float = 0.03,
    noise_row_only: float = 0.05,
    noise_col_only: float = 0.05,
    seed: int = 0
) -> List[List[int]]:
    """
    Próbuje znaleźć rozwiązanie lokalnym przeszukiwaniem.

    Parametry:
    - time_budget_steps: maksymalna liczba iteracji (flipów)
    - restart_after_no_improve: po ilu krokach bez poprawy robimy restart
    - noise_*: prawdopodobieństwa "nieoptymalnych" decyzji
    """
    rng = random.Random(seed)

    best_grid = None
    best_cost = 10**18

    steps = 0
    while steps < time_budget_steps:
        # Restart: losowa plansza (p=0.5 zwykle działa ok)
        grid = random_initial_grid(X, Y, rng, p=0.5)
        row_costs = compute_row_costs(grid, row_desc)
        col_costs = compute_col_costs(grid, col_desc)

        cur_cost = total_cost(row_costs, col_costs)
        last_improve_step = steps

        # Jeśli już trafiliśmy ideał - koniec
        if cur_cost == 0:
            return grid

        # Iteracje w ramach jednego restartu
        while steps < time_budget_steps:
            steps += 1

            # Zapisuj najlepszy stan (na wypadek gdyby nie udało się dojść do 0)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_grid = [row[:] for row in grid]
                last_improve_step = steps

                if best_cost == 0:
                    return best_grid

            # Jeśli długo bez poprawy - restart
            if steps - last_improve_step >= restart_after_no_improve:
                break

            # Budujemy listę "złych" wierszy i kolumn (koszt > 0)
            bad_rows = [i for i in range(X) if row_costs[i] > 0]
            bad_cols = [j for j in range(Y) if col_costs[j] > 0]

            # Jeśli nie ma złych linii - rozwiązanie
            if not bad_rows and not bad_cols:
                return grid

            # Losujemy, czy wybieramy wiersz czy kolumnę (proporcjonalnie do liczby złych)
            choose_row = True
            if bad_rows and bad_cols:
                # losowo, ale z lekkim uwzględnieniem liczebności
                choose_row = (rng.random() < (len(bad_rows) / (len(bad_rows) + len(bad_cols))))
            elif bad_cols:
                choose_row = False

            # Wybór i najlepszy flip
            if choose_row:
                i = rng.choice(bad_rows)
                fi, fj = best_flip_in_row(
                    grid, i, row_desc, col_desc, row_costs, col_costs,
                    rng, noise_random_pixel, noise_row_only
                )
            else:
                j = rng.choice(bad_cols)
                fi, fj = best_flip_in_col(
                    grid, j, row_desc, col_desc, row_costs, col_costs,
                    rng, noise_random_pixel, noise_col_only
                )

            # Zastosuj flip i zaktualizuj koszty
            apply_flip_and_update_costs(grid, fi, fj, row_desc, col_desc, row_costs, col_costs)
            cur_cost = total_cost(row_costs, col_costs)

    # Jeśli nie znaleźliśmy idealnego rozwiązania w budżecie kroków,
    # zwracamy najlepsze znalezione (sprawdzarka zaliczy tylko jeśli było 0,
    # ale w tym zadaniu liczy się liczba rozwiązanych przypadków, więc to i tak najlepsza opcja).
    return best_grid


# ----------------------------
# Main: pliki zgodne z validatorem
# ----------------------------

def main():
    input_file = "zad5_input.txt"
    output_file = "zad5_output.txt"

    X, Y, row_desc, col_desc = read_instance(input_file)

    # Parametry można stroić. Dla małych plansz (np. 7x7) to działa szybko.
    # Dla większych - lepiej zwiększyć budżet kroków i/lub liczbę restartów (tu restart jest w pętli).
    grid = solve_nonogram(
        X, Y, row_desc, col_desc,
        time_budget_steps=250000,
        restart_after_no_improve=7000,
        noise_random_pixel=0.03,
        noise_row_only=0.05,
        noise_col_only=0.05,
        seed=42
    )

    write_solution(output_file, grid)


if __name__ == "__main__":
    main()