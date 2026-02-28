# Zadanie 3 - Figurant (J,Q,K,A) vs Blotkarz (2..10), 5 kart bez dobierania.
#
# Program:
# 1) Symuluje rozdania i szacuje P(wygranej Blotkarza).
# 2) Pozwala zdefiniować talię Blotkarza (dowolny podzbiór blotek).
# 3) Pozwala "wyrzucać" k kart z pełnej talii Blotkarza (heurystyka greedy),
#    żeby znaleźć talię możliwie dużą, która daje Blotkarzowi statystyczną przewagę.
#
# Dlaczego nie implementujemy pełnego porównywania układów?
# - Jeśli obaj gracze mają ten sam typ układu (np. para), to o wyniku decydują rangi/kickery.
# - Figurant ma zawsze wyższe rangi (J,Q,K,A) niż Blotkarz (2..10), więc przy remisie kategorii
#   Figurant zawsze wygrywa.
# - Wystarczy więc porównywać same kategorie układów (rankingi pokerowe).
#
# Uwaga: program jest losowy, więc uruchom kilka razy lub zwiększ liczbę prób.

import random
import sys
from collections import Counter
from typing import List, Tuple, Set, Dict

# Reprezentacja karty: (ranga, kolor)
# ranga: int (2..14), gdzie J=11, Q=12, K=13, A=14
# kolor: jeden znak: 'S','H','D','C'
SUITS = ['S', 'H', 'D', 'C']


# ----------------------------
# Budowa talii
# ----------------------------

def build_deck(ranks: List[int]) -> List[Tuple[int, str]]:
    """Zwraca pełną talię dla podanych rang (wszystkie 4 kolory)."""
    return [(r, s) for r in ranks for s in SUITS]


def figurant_deck() -> List[Tuple[int, str]]:
    """Figurant: tylko J,Q,K,A."""
    return build_deck([11, 12, 13, 14])


def blotkarz_full_deck() -> List[Tuple[int, str]]:
    """Blotkarz: tylko 2..10."""
    return build_deck(list(range(2, 11)))


# ----------------------------
# Ocena układu (tylko KATEGORIA)
# ----------------------------

# Kategorie od najsłabszej do najsilniejszej (wystarczy nam porządek).
# 0: high card
# 1: one pair
# 2: two pair
# 3: three of a kind
# 4: straight
# 5: flush
# 6: full house
# 7: four of a kind
# 8: straight flush
#
# W standardowych zasadach jest jeszcze "royal flush" jako nazwa straight flush do asa,
# ale to i tak ta sama kategoria (straight flush). :contentReference[oaicite:1]{index=1}

def is_straight(ranks_sorted: List[int]) -> bool:
    """Czy 5 rang tworzy strita? (u nas wystarczy wariant klasyczny, bez A2345)"""
    # Dla bezpieczeństwa obsługujemy też A2345 (wheel), chociaż w naszych taliach:
    # - Blotkarz nie ma asa, więc wheel nie zajdzie
    # - Figurant nie ma 2..5, więc wheel też nie zajdzie
    # Ale nie szkodzi mieć poprawnie.
    uniq = sorted(set(ranks_sorted))
    if len(uniq) != 5:
        return False
    # Wheel: A,5,4,3,2
    if uniq == [2, 3, 4, 5, 14]:
        return True
    return max(uniq) - min(uniq) == 4


def hand_category(hand: List[Tuple[int, str]]) -> int:
    """Zwraca kategorię układu 0..8 (nie liczymy kickerów)."""
    ranks = [r for (r, _) in hand]
    suits = [s for (_, s) in hand]
    ranks_sorted = sorted(ranks)

    flush = (len(set(suits)) == 1)
    straight = is_straight(ranks_sorted)

    cnt = Counter(ranks)
    counts = sorted(cnt.values(), reverse=True)  # np. [3,2] dla fulla

    if straight and flush:
        return 8
    if counts == [4, 1]:
        return 7
    if counts == [3, 2]:
        return 6
    if flush:
        return 5
    if straight:
        return 4
    if counts == [3, 1, 1]:
        return 3
    if counts == [2, 2, 1]:
        return 2
    if counts == [2, 1, 1, 1]:
        return 1
    return 0


def blotkarz_wins(blot_hand: List[Tuple[int, str]], fig_hand: List[Tuple[int, str]]) -> bool:
    """
    Porównanie wg zadania:
    - jeśli kategoria Blotkarza > kategoria Figuranta -> Blotkarz wygrywa
    - jeśli równe lub mniejsze -> Blotkarz nie wygrywa (Figurant wygrywa lub remis, ale remis = nie-sukces Blotkarza)
    """
    cb = hand_category(blot_hand)
    cf = hand_category(fig_hand)
    return cb > cf


# ----------------------------
# Symulacja
# ----------------------------

def estimate_win_prob(blot_deck: List[Tuple[int, str]], trials: int, seed: int = 0) -> float:
    """Monte Carlo: zwraca szacowane P(wygranej Blotkarza)."""
    rng = random.Random(seed)

    fdeck = figurant_deck()
    wins = 0

    for _ in range(trials):
        blot_hand = rng.sample(blot_deck, 5)
        fig_hand = rng.sample(fdeck, 5)
        if blotkarz_wins(blot_hand, fig_hand):
            wins += 1

    return wins / trials


# ----------------------------
# Szukanie "dużej" talii dającej przewagę
# ----------------------------

def greedy_remove_to_reach_advantage(
    start_deck: List[Tuple[int, str]],
    trials_eval: int,
    target_prob: float = 0.5,
    max_removals: int = 999,
    seed: int = 123
) -> List[Tuple[int, str]]:
    """
    Heurystyka:
    - Zaczynamy od start_deck (np. pełne blotki 36 kart).
    - Dopóki P < target_prob i mamy budżet usunięć:
      usuwamy 1 kartę tak, żeby P po usunięciu było jak największe.
    - To jest greedy (nie gwarantuje optimum), ale często działa dobrze.
    """
    rng = random.Random(seed)
    deck = start_deck[:]

    # Małe przyspieszenie: losujemy kolejność testowania usunięć, żeby nie było biasu.
    while len(deck) > 5 and max_removals > 0:
        p = estimate_win_prob(deck, trials_eval, seed=rng.randrange(10**9))
        if p >= target_prob:
            break

        best_p = -1.0
        best_idx = None

        indices = list(range(len(deck)))
        rng.shuffle(indices)

        for i in indices:
            candidate = deck[:i] + deck[i+1:]
            if len(candidate) < 5:
                continue
            cp = estimate_win_prob(candidate, trials_eval, seed=rng.randrange(10**9))
            if cp > best_p:
                best_p = cp
                best_idx = i

        if best_idx is None:
            break

        deck.pop(best_idx)
        max_removals -= 1

    return deck


def greedy_add_back_maximize_size(
    base_deck: List[Tuple[int, str]],
    removed_cards: List[Tuple[int, str]],
    trials_eval: int,
    target_prob: float = 0.5,
    seed: int = 777
) -> List[Tuple[int, str]]:
    """
    Druga faza:
    - Mamy deck, który już spełnia P>=target_prob.
    - Próbujemy dodawać z powrotem karty tak, żeby maksymalizować rozmiar,
      ale nie zejść poniżej target_prob.
    - Też greedy: zawsze dodajemy taką kartę, która najmniej psuje P (albo najbardziej poprawia).
    """
    rng = random.Random(seed)
    deck = base_deck[:]
    pool = removed_cards[:]

    while pool:
        best_p = -1.0
        best_j = None

        # losujemy kolejność, żeby nie faworyzować konkretnych kart
        js = list(range(len(pool)))
        rng.shuffle(js)

        for j in js:
            candidate = deck + [pool[j]]
            cp = estimate_win_prob(candidate, trials_eval, seed=rng.randrange(10**9))
            if cp >= target_prob and cp > best_p:
                best_p = cp
                best_j = j

        if best_j is None:
            break

        deck.append(pool.pop(best_j))

    return deck


def find_large_advantage_deck(
    trials_eval: int = 8000,
    trials_final: int = 60000,
    target_prob: float = 0.5
) -> None:
    """
    Kompletny eksperyment:
    1) Liczymy P dla pełnej talii blotek (36).
    2) Greedy usuwamy karty, aż osiągniemy przewagę.
    3) Potem greedy dodajemy z powrotem, żeby zwiększyć rozmiar talii.
    4) Na końcu robimy dokładniejsze oszacowanie na większej liczbie prób.
    """
    full = blotkarz_full_deck()

    p_full = estimate_win_prob(full, trials_final, seed=1)
    print(f"Pełna talia Blotkarza: {len(full)} kart, P(win) ~ {p_full:.4f}")

    # Faza usuwania
    deck_after_remove = greedy_remove_to_reach_advantage(
        start_deck=full,
        trials_eval=trials_eval,
        target_prob=target_prob,
        max_removals=1000,
        seed=2
    )

    removed = [c for c in full if c not in deck_after_remove]
    p_mid = estimate_win_prob(deck_after_remove, trials_final, seed=3)
    print(f"Po usuwaniu: {len(deck_after_remove)} kart, P(win) ~ {p_mid:.4f}")

    # Faza dodawania
    deck_best = greedy_add_back_maximize_size(
        base_deck=deck_after_remove,
        removed_cards=removed,
        trials_eval=trials_eval,
        target_prob=target_prob,
        seed=4
    )

    p_best = estimate_win_prob(deck_best, trials_final, seed=5)
    print(f"Po dodawaniu: {len(deck_best)} kart, P(win) ~ {p_best:.4f}")

    # Wypisz proponowaną talię (rangi + kolory)
    # To ma być czytelne: np. "2S 2H ... 10D"
    def fmt(card):
        r, s = card
        rank_str = {11: "J", 12: "Q", 13: "K", 14: "A"}.get(r, str(r))
        return f"{rank_str}{s}"

    print("Proponowana talia Blotkarza:")
    print(" ".join(fmt(c) for c in sorted(deck_best, key=lambda x: (x[1], x[0]))))


# ----------------------------
# Proste, ręczne eksperymenty opisane w treści zadania
# ----------------------------

def experiment_fixed_decks(trials: int = 60000):
    """
    Pokazuje kilka sensownych talii "z ręki", żebyś widział kierunek:
    - pełna talia blotek (36)
    - jedna pełna barwa (9) -> zawsze kolor, Figurant nie może mieć koloru
    - dwie pełne barwy (18)
    """
    full = blotkarz_full_deck()
    one_suit = [(r, 'S') for r in range(2, 11)]  # 9 kart, zawsze kolor
    two_suits = [(r, s) for r in range(2, 11) for s in ['S', 'H']]  # 18 kart

    print(f"FULL 36:  P(win) ~ {estimate_win_prob(full, trials, seed=10):.4f}")
    print(f"1 SUIT 9: P(win) ~ {estimate_win_prob(one_suit, trials, seed=11):.4f}")
    print(f"2 SUITS 18:P(win) ~ {estimate_win_prob(two_suits, trials, seed=12):.4f}")


# ----------------------------
# CLI
# ----------------------------

def main():
    # Przykłady użycia:
    # - python zad3.py --exp
    # - python zad3.py --search
    # - python zad3.py --prob trials
    #
    # Domyślnie: uruchamia wyszukiwanie dużej talii z przewagą.

    if len(sys.argv) >= 2 and sys.argv[1] == "--exp":
        experiment_fixed_decks(trials=60000)
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "--search":
        find_large_advantage_deck(trials_eval=8000, trials_final=60000, target_prob=0.5)
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--prob":
        trials = int(sys.argv[2])
        p = estimate_win_prob(blotkarz_full_deck(), trials, seed=42)
        print(f"Pełna talia Blotkarza (36), P(win) ~ {p:.4f}")
        return

    # Domyślnie szukamy talii możliwie dużej z przewagą.
    find_large_advantage_deck(trials_eval=8000, trials_final=60000, target_prob=0.5)


if __name__ == "__main__":
    main()