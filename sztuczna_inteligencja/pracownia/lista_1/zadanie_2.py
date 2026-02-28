# Zadanie 2 - rekonstrukcja tekstu bez spacji (DP + słownik).
#
# Cel:
# Dla danego tekstu t (bez spacji) chcemy wstawić spacje tak, aby:
# 1) po usunięciu spacji wrócić do oryginalnego t
# 2) każde słowo należy do słownika S (polish_words.txt)
# 3) maksymalizujemy sumę kwadratów długości słów (preferujemy długie słowa)
#
# Kluczowy algorytm:
# - Używamy programowania dynamicznego (DP) po pozycjach w tekście.
# - dp[i] = najlepszy wynik (maks. suma kwadratów) dla prefiksu t[0:i]
# - parent[i] = (j, word) informacja skąd przyszliśmy: ostatnie słowo to t[j:i]
# - Przejścia: dla każdego i próbujemy dopasować wszystkie słowa zaczynające się w i
#   (czyli t[i:k]) i aktualizujemy dp[k].
#
# Żeby szybko znajdować wszystkie słowa zaczynające się w pozycji i, budujemy TRIE:
# - Przechodzimy po znakach t[i], t[i+1], ... schodząc w trie
# - Za każdym razem gdy trafimy w węzeł kończący słowo, mamy kandydackie słowo.
#
# Złożoność:
# - Bez trie: DP wymagałoby sprawdzania wielu substringów w set (koszt tworzenia substringów).
# - Z trie: dla każdej pozycji idziemy maksymalnie do długości najdłuższego słowa w słowniku,
#   ale kończymy wcześniej jeśli nie ma ścieżki w trie.
#
# Jeśli nie da się zrekonstruować (brak podziału), wypisujemy "INF" (bezpieczny fallback).

import sys
from typing import Dict, Optional, Tuple, List


# ----------------------------
# TRIE - struktura do dopasowań prefiksowych
# ----------------------------

class TrieNode:
    __slots__ = ("children", "is_word")
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.max_word_len = 0  # pomaga ograniczyć skanowanie po tekście

    def insert(self, word: str):
        node = self.root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = TrieNode()
                node.children[ch] = nxt
            node = nxt
        node.is_word = True
        if len(word) > self.max_word_len:
            self.max_word_len = len(word)

    def iter_matches_from(self, text: str, start: int):
        """
        Generator zwracający wszystkie (end_index, matched_word) takie, że:
        matched_word = text[start:end_index] i należy do słownika.
        end_index jest "po końcu" (jak w Python slicing).
        """
        node = self.root
        # Idziemy od start w prawo, ale nie dalej niż max_word_len
        limit = min(len(text), start + self.max_word_len)

        for i in range(start, limit):
            ch = text[i]
            node = node.children.get(ch)
            if node is None:
                return  # dalej już nie dopasujemy niczego
            if node.is_word:
                # Mamy słowo od start do i włącznie, czyli end = i+1
                yield (i + 1, text[start:i + 1])


# ----------------------------
# Wczytywanie słownika
# ----------------------------

def load_dictionary_trie(path: str) -> Trie:
    """
    Wczytuje polish_words.txt (UTF-8) i buduje trie.
    Zakładamy, że w pliku jest 1 słowo na linię.
    """
    trie = Trie()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            trie.insert(w)
    return trie


# ----------------------------
# DP rekonstrukcji dla jednej linii
# ----------------------------

def reconstruct_best(text: str, trie: Trie) -> str:
    """
    Zwraca tekst ze spacjami dla najlepszego podziału wg sumy kwadratów długości.
    Jeśli się nie da, zwraca "INF".
    """
    n = len(text)

    # dp[i] = najlepsza suma kwadratów dla prefiksu text[0:i]
    # Używamy -1 jako "nieosiągalne"
    dp = [-1] * (n + 1)
    dp[0] = 0

    # parent[i] = (j, word) czyli ostatnie słowo to text[j:i]
    parent: List[Optional[Tuple[int, str]]] = [None] * (n + 1)

    # Przechodzimy po wszystkich pozycjach startowych i próbujemy dopasować słowa
    for i in range(n):
        if dp[i] < 0:
            continue  # prefiks do i jest nieosiągalny, nie ma sensu rozwijać

        # Znajdź wszystkie słowa zaczynające się w i
        for end, word in trie.iter_matches_from(text, i):
            score = dp[i] + (len(word) * len(word))

            # Jeśli znaleźliśmy lepszy wynik dla dp[end], aktualizujemy
            if score > dp[end]:
                dp[end] = score
                parent[end] = (i, word)

    # Jeśli dp[n] nieosiągalne, nie ma poprawnego podziału
    if dp[n] < 0:
        return "INF"

    # Odtwarzanie rozwiązania z parent[]
    words = []
    cur = n
    while cur > 0:
        p = parent[cur]
        if p is None:
            # Teoretycznie nie powinno się zdarzyć, skoro dp[n] jest osiągalne,
            # ale zostawiamy zabezpieczenie.
            return "INF"
        prev, w = p
        words.append(w)
        cur = prev

    words.reverse()
    return " ".join(words)


# ----------------------------
# Tryb wsadowy: pliki wejścia/wyjścia
# ----------------------------

def batch_mode(dict_path="polish_words.txt",
               in_path="zad2_input.txt",
               out_path="zad2_output.txt"):
    trie = load_dictionary_trie(dict_path)

    out_lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            # Usuwamy tylko znak końca linii. Tekst nie ma spacji, ale gdyby miał,
            # to zadanie mówi, że wejście jest bez spacji, więc traktujemy to jako dane.
            text = line.rstrip("\n")
            if text == "":
                out_lines.append("")  # pusta linia -> pusta linia
                continue
            out_lines.append(reconstruct_best(text, trie))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


# ----------------------------
# Debug: szybki test pojedynczego napisu z argumentu
# ----------------------------

def debug_mode():
    if len(sys.argv) < 3:
        print("Użycie: python zad2.py --debug <tekst_bez_spacji>")
        return
    text = sys.argv[2]
    trie = load_dictionary_trie("polish_words.txt")
    print(reconstruct_best(text, trie))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--debug":
        debug_mode()
    else:
        batch_mode()


if __name__ == "__main__":
    main()