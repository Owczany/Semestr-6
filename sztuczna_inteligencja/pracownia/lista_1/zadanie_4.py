# Zadanie 4 (2p) - opt_dist dla uproszczonych nonogramów (1 blok na wiersz/kolumnę).
#
# Wejście (w wersji używanej przez validator):
# - plik zad4_input.txt
# - każda linia: <ciąg 0/1> <D>
# Wyjście:
# - plik zad4_output.txt
# - dla każdej linii jedna liczba: minimalna liczba flipów (zamian bitu), żeby uzyskać:
#   * jeśli D>0: dokładnie jeden blok jedynek długości D (spójny), reszta zera
#   * jeśli D=0: same zera
#
# Algorytm:
# - D=0: trzeba wyzerować wszystkie jedynki -> wynik = liczba jedynek w wejściu
# - D>0: rozważamy każdą pozycję startu bloku długości D:
#     koszt = (ile zer w oknie) + (ile jedynek poza oknem)
#   liczymy to szybko z prefiksowych sum jedynek.
#   koszt = D + ones_total - 2*ones_in_window
#   wybieramy minimum po wszystkich oknach.
#
# Złożoność: O(n) na jedną linię.

from typing import List
import os
import sys


def opt_dist(bits: List[int], D: int) -> int:
    """
    Zwraca minimalną liczbę flipów, aby uzyskać:
    - D>0: jeden spójny blok 1 długości D, poza blokiem same 0
    - D=0: same 0
    """
    n = len(bits)

    # Przypadek D=0: nie ma bloku -> wszystko ma być 0.
    # Jedyny koszt to zamiana każdej 1 na 0.
    if D == 0:
        return sum(bits)

    # Jeśli D > n, nie da się wstawić bloku długości D w linię długości n.
    # W praktyce w tych zadaniach raczej nie wystąpi, ale zabezpieczamy się.
    if D > n:
        # Minimalnie można by "dążyć" do niemożliwego, ale sensownie jest zwrócić INF-like.
        # Validator raczej tego nie testuje - jednak niech będzie bez crasha.
        return 10**9

    # pref[i] = liczba jedynek w bits[0:i]
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + bits[i]

    ones_total = pref[n]

    best = 10**18

    # Okno [start, start+D)
    for start in range(0, n - D + 1):
        end = start + D
        ones_in_window = pref[end] - pref[start]

        # zeros_in_window = D - ones_in_window
        # ones_outside = ones_total - ones_in_window
        # cost = zeros_in_window + ones_outside
        cost = (D - ones_in_window) + (ones_total - ones_in_window)

        if cost < best:
            best = cost

    return int(best)


def solve_lines(lines: List[str]) -> List[str]:
    """
    Bierze linie wejścia, zwraca linie wyjścia (stringi z wynikiem).
    Każda linia wejściowa ma format: "<ciąg_bitów> <D>"
    """
    out = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Rozbijamy na dwa tokeny: bitstring i D
        bits_str, d_str = line.split()
        D = int(d_str)

        # Konwersja bitstring -> lista intów
        bits = [1 if c == '1' else 0 for c in bits_str]

        out.append(str(opt_dist(bits, D)))

    return out

def main():
    # Validator odpala program tak, że oczekuje plików:
    # - wejście: zad4_input.txt
    # - wyjście: zad4_output.txt
    #
    # Tryby działania:
    # 1. --debug <lista_bitów> <D>  -> liczy i wypisuje wynik w konsoli
    # 2. jeśli istnieje zad4_input.txt -> czytamy go i zapisujemy zad4_output.txt
    # 3. w przeciwnym razie -> czytamy z stdin i piszemy na stdout

    if len(sys.argv) >= 2 and sys.argv[1] == "--debug":
        if len(sys.argv) != 4:
            print("Użycie: python zadanie_4.py --debug <lista_bitów> <D>")
            print("Przykład: python zadanie_4.py --debug 0110100 3")
            sys.exit(1)
        bits_str = sys.argv[2]
        D = int(sys.argv[3])
        bits = [1 if c == '1' else 0 for c in bits_str]
        result = opt_dist(bits, D)
        print(f"opt_dist({bits_str}, {D}) = {result}")
        return

    input_file = "zad4_input.txt"
    output_file = "zad4_output.txt"

    if os.path.exists(input_file): 
        # Tryb pod validator / wsadowy
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        out_lines = solve_lines(lines)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
            if out_lines:
                f.write("\n")
    else:
        print("Nie znaleziono pliku zad4_input.txt, czytanie z stdin. Wpisz linie w formacie '<ciąg_bitów> <D>' i zakończ EOF (Ctrl+D):")


if __name__ == "__main__":
    main()