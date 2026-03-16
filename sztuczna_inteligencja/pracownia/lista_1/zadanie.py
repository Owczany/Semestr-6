# Zadanie 2 + algorytm losowy do porównania rekonstrukcji.
#
# Tryby:
# 1) domyślny:
#    python zad2.py
#    czyta zad2_input.txt i zapisuje najlepszą rekonstrukcję do zad2_output.txt
#
# 2) debug:
#    python zad2.py --debug <tekst_bez_spacji>
#
# 3) losowy:
#    python zad2.py --random
#    czyta zad2_input.txt i zapisuje losową rekonstrukcję do random_out.txt
#
# 4) losowy z seedem:
#    python zad2.py --random --seed 123
#
# Idea algorytmu losowego:
# - dla każdej pozycji znajdujemy wszystkie słowa słownikowe zaczynające się w tej pozycji
# - losowo wybieramy jedno z nich
# - aby nie wpadać bez sensu w ślepe zaułki, wcześniej liczymy tablicę reachable[i],
#   która mówi, czy z pozycji i da się dojść do końca poprawnym podziałem
# - losujemy tylko spośród słów prowadzących do pozycji "osiągalnych"

import sys
import random
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
        self.max_word_len = 0

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
        limit = min(len(text), start + self.max_word_len)

        for i in range(start, limit):
            ch = text[i]
            node = node.children.get(ch)
            if node is None:
                return
            if node.is_word:
                yield (i + 1, text[start:i + 1])


# ----------------------------
# Wczytywanie słownika
# ----------------------------

def load_dictionary_trie(path: str) -> Trie:
    trie = Trie()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            trie.insert(w)
    return trie


# ----------------------------
# DP rekonstrukcji dla jednej linii - najlepsza
# ----------------------------

def reconstruct_best(text: str, trie: Trie) -> str:
    n = len(text)

    dp = [-1] * (n + 1)
    dp[0] = 0

    parent: List[Optional[Tuple[int, str]]] = [None] * (n + 1)

    for i in range(n):
        if dp[i] < 0:
            continue

        for end, word in trie.iter_matches_from(text, i):
            score = dp[i] + len(word) * len(word)

            if score > dp[end]:
                dp[end] = score
                parent[end] = (i, word)

    if dp[n] < 0:
        return "INF"

    words = []
    cur = n
    while cur > 0:
        p = parent[cur]
        if p is None:
            return "INF"
        prev, w = p
        words.append(w)
        cur = prev

    words.reverse()
    return " ".join(words)


# ----------------------------
# Pomocnicze: osiągalność końca
# ----------------------------

def compute_reachable(text: str, trie: Trie) -> List[bool]:
    """
    reachable[i] = czy z pozycji i da się dojść do końca tekstu,
    dzieląc suffix text[i:] na słowa ze słownika.
    """
    n = len(text)
    reachable = [False] * (n + 1)
    reachable[n] = True

    for i in range(n - 1, -1, -1):
        for end, _word in trie.iter_matches_from(text, i):
            if reachable[end]:
                reachable[i] = True
                break

    return reachable


# ----------------------------
# Rekonstrukcja losowa
# ----------------------------

def reconstruct_random(text: str, trie: Trie, rng: random.Random) -> str:
    """
    Losowa rekonstrukcja:
    - na każdej pozycji losujemy jedno z pasujących słów,
      ale tylko takie, po którym da się jeszcze dojść do końca.
    - jeśli cały napis jest nierozkładalny, zwracamy "INF"
    """
    n = len(text)
    reachable = compute_reachable(text, trie)

    if not reachable[0]:
        return "INF"

    words = []
    pos = 0

    while pos < n:
        candidates = []
        for end, word in trie.iter_matches_from(text, pos):
            if reachable[end]:
                candidates.append((end, word))

        if not candidates:
            return "INF"

        end, word = rng.choice(candidates)
        words.append(word)
        pos = end

    return " ".join(words)


# ----------------------------
# Tryb wsadowy - najlepsza rekonstrukcja
# ----------------------------

def batch_mode(dict_path="polish_words.txt",
               in_path="zad2_input.txt",
               out_path="zad2_output.txt"):
    trie = load_dictionary_trie(dict_path)

    out_lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            if text == "":
                out_lines.append("")
                continue
            out_lines.append(reconstruct_best(text, trie))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


# ----------------------------
# Tryb wsadowy - losowa rekonstrukcja
# ----------------------------

def random_batch_mode(dict_path="polish_words.txt",
                      in_path="zad2_input.txt",
                      out_path="random_out.txt",
                      seed: Optional[int] = None):
    trie = load_dictionary_trie(dict_path)
    rng = random.Random(seed)

    out_lines = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            if text == "":
                out_lines.append("")
                continue
            out_lines.append(reconstruct_random(text, trie, rng))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


# ----------------------------
# Debug: szybki test pojedynczego napisu
# ----------------------------

def debug_mode():
    if len(sys.argv) < 3:
        print("Użycie: python zad2.py --debug <tekst_bez_spacji>")
        return
    text = sys.argv[2]
    trie = load_dictionary_trie("polish_words.txt")
    print(reconstruct_best(text, trie))


def debug_random_mode(seed: Optional[int] = None):
    if len(sys.argv) < 3:
        print("Użycie: python zad2.py --random-debug <tekst_bez_spacji> [--seed N]")
        return
    text = sys.argv[2]
    trie = load_dictionary_trie("polish_words.txt")
    rng = random.Random(seed)
    print(reconstruct_random(text, trie, rng))


# ----------------------------
# Parsowanie prostych argumentów
# ----------------------------

def parse_seed(argv: List[str]) -> Optional[int]:
    if "--seed" in argv:
        idx = argv.index("--seed")
        if idx + 1 >= len(argv):
            raise ValueError("Po --seed musi wystąpić liczba całkowita.")
        return int(argv[idx + 1])
    return None


def main():
    try:
        seed = parse_seed(sys.argv[1:])
    except ValueError as e:
        print(f"Błąd argumentów: {e}")
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "--debug":
        debug_mode()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--random":
        random_batch_mode(seed=seed)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--random-debug":
        debug_random_mode(seed=seed)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--normal":
        batch_mode()
    else:
        # TODO: Dopisać funkcję, która robi analizę algorytmu z dp i losowego, i z pliku good answer, liczy zgodność i wypisuje statystyki. Nazwa pliku z wzorcową rekonstrukcją nazywa się "good_answer.txt".



if __name__ == "__main__":
    main()