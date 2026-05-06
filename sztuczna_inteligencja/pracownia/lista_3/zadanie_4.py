import os
import sys


def V(i, j):
    return "V%d_%d" % (i, j)


def read_input():
    if os.path.exists("zad_input.txt"):
        with open("zad_input.txt", "r", encoding="utf-8") as f:
            data = f.read()
        return data, True

    return sys.stdin.read(), False


def parse(data):
    lines = [line.strip() for line in data.splitlines() if line.strip()]
    if not lines:
        return [], [], []

    rows = [int(x) for x in lines[0].split()]
    cols = [int(x) for x in lines[1].split()]
    triples = [tuple(int(x) for x in line.split()) for line in lines[2:]]
    return rows, cols, triples


def print_constraints(constraints, indent=4, width=70):
    position = indent
    print(indent * " ", end="")

    for constraint in constraints:
        print(constraint + ",", end=" ")
        position += len(constraint) + 2
        if position > width:
            position = indent
            print()
            print(indent * " ", end="")


def domains(variables):
    return [variable + " in 0..1" for variable in variables]


def row_sums(rows, n, m):
    constraints = []
    for i in range(n):
        variables = [V(i, j) for j in range(m)]
        constraints.append("sum([" + ", ".join(variables) + "], #=, %d)" % rows[i])
    return constraints


def column_sums(cols, n, m):
    constraints = []
    for j in range(m):
        variables = [V(i, j) for i in range(n)]
        constraints.append("sum([" + ", ".join(variables) + "], #=, %d)" % cols[j])
    return constraints


def known_fields(triples):
    return ["%s #= %d" % (V(i, j), value) for i, j, value in triples]


def square_constraints(n, m):
    constraints = []
    for i in range(n - 1):
        for j in range(m - 1):
            constraints.append(
                "%s + %s #= 2 #<==> %s + %s #= 2"
                % (V(i, j), V(i + 1, j + 1), V(i + 1, j), V(i, j + 1))
            )
    return constraints


def triple_constraints(n, m):
    constraints = []

    for i in range(n):
        for j in range(m - 2):
            constraints.append(
                "%s #= 1 #==> %s + %s #> 0"
                % (V(i, j + 1), V(i, j), V(i, j + 2))
            )

    for i in range(n - 2):
        for j in range(m):
            constraints.append(
                "%s #= 1 #==> %s + %s #> 0"
                % (V(i + 1, j), V(i, j), V(i + 2, j))
            )

    return constraints


def storms(rows, cols, triples):
    n = len(rows)
    m = len(cols)
    variables = [V(i, j) for i in range(n) for j in range(m)]

    constraints = (
        domains(variables)
        + row_sums(rows, n, m)
        + column_sums(cols, n, m)
        + known_fields(triples)
        + square_constraints(n, m)
        + triple_constraints(n, m)
    )

    print(":- use_module(library(clpfd)).")
    print("solve([" + ", ".join(variables) + "]) :- ")
    print_constraints(constraints)
    print()
    print("    labeling([ff], [" + ", ".join(variables) + "]).")
    print()
    print(":- solve(X), write(X), nl.")


def build_output(data):
    rows, cols, triples = parse(data)
    if not rows and not cols:
        return ""

    original_stdout = sys.stdout
    try:
        from io import StringIO

        buffer = StringIO()
        sys.stdout = buffer
        storms(rows, cols, triples)
        return buffer.getvalue()
    finally:
        sys.stdout = original_stdout


def main():
    data, use_files = read_input()
    result = build_output(data)

    if use_files:
        with open("zad_output.txt", "w", encoding="utf-8") as f:
            f.write(result)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
