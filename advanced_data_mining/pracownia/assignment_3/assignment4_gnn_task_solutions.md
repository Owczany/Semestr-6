# Assignment 4 - GNN
## Detailed solutions and explanations

Based on the assignment sheet. The original tasks are in the uploaded PDF. 

---

## Task 1: Action of the Graph Laplacian

We consider the path graph

```text
1 - 2 - 3
```

and the vector

\[
x = \begin{bmatrix}1\\0\\0\end{bmatrix}.
\]

The graph is undirected and unweighted.

### 1. Compute the matrices \(A\), \(D\), \(L=D-A\)

#### Adjacency matrix \(A\)
Vertex 1 is connected only to 2, vertex 2 is connected to 1 and 3, and vertex 3 is connected only to 2. Therefore

\[
A=
\begin{bmatrix}
0 & 1 & 0\\
1 & 0 & 1\\
0 & 1 & 0
\end{bmatrix}.
\]

#### Degree matrix \(D\)
The degrees are:
- \(\deg(1)=1\)
- \(\deg(2)=2\)
- \(\deg(3)=1\)

Hence

\[
D=
\begin{bmatrix}
1 & 0 & 0\\
0 & 2 & 0\\
0 & 0 & 1
\end{bmatrix}.
\]

#### Laplacian \(L=D-A\)
Subtracting the matrices gives

\[
L=
\begin{bmatrix}
1 & -1 & 0\\
-1 & 2 & -1\\
0 & -1 & 1
\end{bmatrix}.
\]

---

### 2. Compute \(Lx\)

We can compute this either by matrix multiplication or by using the formula

\[
(Lx)_i = \sum_{j\sim i}(x_i-x_j).
\]

Let us do it coordinate by coordinate.

#### Vertex 1
Vertex 1 has only one neighbor, namely 2. Thus

\[
(Lx)_1 = x_1-x_2 = 1-0 = 1.
\]

#### Vertex 2
Vertex 2 has neighbors 1 and 3. Thus

\[
(Lx)_2 = (x_2-x_1) + (x_2-x_3) = (0-1)+(0-0) = -1.
\]

#### Vertex 3
Vertex 3 has only one neighbor, namely 2. Thus

\[
(Lx)_3 = x_3-x_2 = 0-0 = 0.
\]

Therefore

\[
Lx=
\begin{bmatrix}
1\\
-1\\
0
\end{bmatrix}.
\]

#### Which vertices have non-zero values?
The non-zero entries appear at vertices 1 and 2.

#### On which values does \((Lx)_i\) depend?
The value \((Lx)_i\) depends only on:
- the value at vertex \(i\), that is \(x_i\), and
- the values at the neighbors of vertex \(i\).

So Laplacian action is local: it uses only one-hop neighborhood information.

#### Describe in words what operation \(Lx\) performs
The vector \(Lx\) measures how different the value at each vertex is from the values at its neighbors. If a vertex has a larger value than its neighbors, the result tends to be positive there. If it has a smaller value than its neighbors, the result tends to be negative. If it matches its neighborhood, the result tends to be zero.

So the graph Laplacian acts like a discrete difference operator on the graph.

---

### 3. Compute \(L^2x = L(Lx)\)

We already found

\[
Lx=
\begin{bmatrix}
1\\
-1\\
0
\end{bmatrix}.
\]

Now apply \(L\) once more.

Let

\[
y=Lx=
\begin{bmatrix}
1\\
-1\\
0
\end{bmatrix}.
\]

Then

#### Vertex 1
\[
(Ly)_1 = y_1-y_2 = 1-(-1)=2.
\]

#### Vertex 2
\[
(Ly)_2 = (y_2-y_1)+(y_2-y_3)=(-1-1)+(-1-0)=-2-1=-3.
\]

#### Vertex 3
\[
(Ly)_3 = y_3-y_2 = 0-(-1)=1.
\]

Hence

\[
L^2x=
\begin{bmatrix}
2\\
-3\\
1
\end{bmatrix}.
\]

#### Does a non-zero value appear at vertex 3?
Yes. In \(Lx\), vertex 3 had value 0, but in \(L^2x\) the value at vertex 3 becomes 1.

#### Does \(L^2x\) depend only on neighbors, or also on more distant vertices?
It depends also on more distant vertices. More precisely:
- \(Lx\) depends on 1-hop neighbors,
- \(L^2x\) depends on information that can propagate through 2 hops.

In this example, the initial signal at vertex 1 influences vertex 3 after applying \(L\) twice.

---

### 4. General rule describing the behavior of \(L^k x\)

Each application of \(L\) uses only the current vertex and its neighbors. Therefore, after applying \(L\) repeatedly \(k\) times, the value at a vertex can depend on vertices up to distance at most \(k\).

So the general rule is:

> \(L^k x\) propagates information through the graph up to the \(k\)-hop neighborhood.

Important clarification: this is not just direct summation of values from all vertices at distance at most \(k\). Instead, it is the result of repeatedly applying a difference operator. Thus \(L^k x\) captures more and more global structure, but through iterated local differences.

---

## Task 2: Weighted Graph Laplacian

We consider the weighted graph

```text
1 -(1)- 2 -(2)- 3
```

with weights
- \(w_{12}=1\)
- \(w_{23}=2\)

and the vector

\[
x = \begin{bmatrix}1\\0\\0\end{bmatrix}.
\]

---

### 1. Construct the matrices \(W\), \(D\), \(L=D-W\)

#### Weighted adjacency matrix \(W\)
The graph is undirected, so the matrix is symmetric:

\[
W=
\begin{bmatrix}
0 & 1 & 0\\
1 & 0 & 2\\
0 & 2 & 0
\end{bmatrix}.
\]

#### Degree matrix \(D\)
In the weighted case, the degree of a vertex is the sum of weights incident to that vertex.

- Vertex 1: \(1\)
- Vertex 2: \(1+2=3\)
- Vertex 3: \(2\)

Hence

\[
D=
\begin{bmatrix}
1 & 0 & 0\\
0 & 3 & 0\\
0 & 0 & 2
\end{bmatrix}.
\]

#### Weighted Laplacian \(L=D-W\)
Therefore

\[
L=
\begin{bmatrix}
1 & -1 & 0\\
-1 & 3 & -2\\
0 & -2 & 2
\end{bmatrix}.
\]

---

### 2. Compute \(Lx\)

For a weighted graph,

\[
(Lx)_i = \sum_{j\sim i} w_{ij}(x_i-x_j).
\]

#### Vertex 1
\[
(Lx)_1 = 1\cdot(1-0)=1.
\]

#### Vertex 2
\[
(Lx)_2 = 1\cdot(0-1)+2\cdot(0-0)=-1+0=-1.
\]

#### Vertex 3
\[
(Lx)_3 = 2\cdot(0-0)=0.
\]

Thus

\[
Lx=
\begin{bmatrix}
1\\
-1\\
0
\end{bmatrix}.
\]

#### Compare with the unweighted case
The result is exactly the same as in Task 1.

Why? Because the only non-zero difference in the initial vector is between vertices 1 and 2. The edge between 2 and 3 does not contribute at this stage, because \(x_2=x_3=0\), hence \(x_2-x_3=0\).

#### Which influence is stronger: from vertex 1 to 2, or from 2 to 3?
In the graph itself, the influence from 2 to 3 is stronger, because its edge weight is larger:

\[
w_{23}=2 > w_{12}=1.
\]

However, this stronger influence does not yet show up in \(Lx\), because the values at 2 and 3 are initially equal.

---

### 3. Compute \(L^2x\)

We already have

\[
Lx=
\begin{bmatrix}
1\\
-1\\
0
\end{bmatrix}.
\]

Let

\[
y=Lx=
\begin{bmatrix}
1\\
-1\\
0
\end{bmatrix}.
\]

Now apply the weighted Laplacian again.

#### Vertex 1
\[
(Ly)_1 = 1\cdot(1-(-1))=2.
\]

#### Vertex 2
\[
(Ly)_2 = 1\cdot((-1)-1)+2\cdot((-1)-0)=-2-2=-4.
\]

#### Vertex 3
\[
(Ly)_3 = 2\cdot(0-(-1))=2.
\]

So

\[
L^2x=
\begin{bmatrix}
2\\
-4\\
2
\end{bmatrix}.
\]

#### Does the value at vertex 3 depend on the weight \(w_{23}\)? Explain.
Yes. The third coordinate is

\[
(L^2x)_3 = w_{23}(y_3-y_2).
\]

Since \(y_3=0\) and \(y_2=-1\), we get

\[
(L^2x)_3 = w_{23}(0-(-1)) = w_{23}.
\]

So the value at vertex 3 depends directly on \(w_{23}\). In our case, because \(w_{23}=2\), the value becomes 2. If the weight were larger, the value would be larger as well.

---

### 4. Show that

\[
x^T L x = \sum_{(i,j)\in E} w_{ij}(x_i-x_j)^2.
\]

We prove this step by step.

Since \(L=D-W\),

\[
x^T L x = x^T(D-W)x = x^T D x - x^T W x.
\]

#### First term: \(x^T D x\)
Because \(D\) is diagonal,

\[
x^T D x = \sum_i D_{ii}x_i^2.
\]

In the weighted graph,

\[
D_{ii} = \sum_j w_{ij}.
\]

Therefore

\[
x^T D x = \sum_i \left(\sum_j w_{ij}\right)x_i^2
= \sum_{i,j} w_{ij}x_i^2.
\]

#### Second term: \(x^T W x\)
By matrix multiplication,

\[
x^T W x = \sum_{i,j} w_{ij}x_ix_j.
\]

#### Subtracting the two terms
So

\[
x^T L x = \sum_{i,j} w_{ij}x_i^2 - \sum_{i,j} w_{ij}x_ix_j
= \sum_{i,j} w_{ij}(x_i^2-x_ix_j).
\]

Now use the symmetry of the undirected graph: \(w_{ij}=w_{ji}\). This lets us combine the terms for \((i,j)\) and \((j,i)\). Then

\[
\sum_{i,j} w_{ij}(x_i^2-x_ix_j)
= \frac{1}{2}\sum_{i,j} w_{ij}(x_i^2-2x_ix_j+x_j^2).
\]

But

\[
x_i^2-2x_ix_j+x_j^2 = (x_i-x_j)^2.
\]

Hence

\[
x^T L x = \frac{1}{2}\sum_{i,j} w_{ij}(x_i-x_j)^2.
\]

The factor \(\tfrac12\) appears because in the double sum over \(i,j\), each undirected edge is counted twice: once as \((i,j)\) and once as \((j,i)\). Therefore we can rewrite the expression as a sum over edges, counted only once:

\[
x^T L x = \sum_{(i,j)\in E} w_{ij}(x_i-x_j)^2.
\]

This proves the identity.

#### Verification for the concrete vector
For our vector \(x=[1,0,0]^T\):

- Edge \((1,2)\): contribution is \(1\cdot(1-0)^2=1\)
- Edge \((2,3)\): contribution is \(2\cdot(0-0)^2=0\)

So the right-hand side equals 1.

On the other hand,

\[
Lx=
\begin{bmatrix}1\\-1\\0\end{bmatrix},
\quad
x^TLx = [1,0,0]
\begin{bmatrix}1\\-1\\0\end{bmatrix}=1.
\]

So both sides agree.

---

### 5. Interpret the role of weights in the expression above. What happens when a weight \(w_{ij}\) is very large?

The expression

\[
x^T L x = \sum_{(i,j)\in E} w_{ij}(x_i-x_j)^2
\]

measures the total weighted variation of the signal \(x\) across graph edges.

- If two adjacent vertices have very different values, then \((x_i-x_j)^2\) is large.
- If the edge between them has large weight, that difference is penalized more strongly.

So the weight \(w_{ij}\) controls how strongly the graph prefers the values at vertices \(i\) and \(j\) to be similar.

If \(w_{ij}\) is very large, then any difference between \(x_i\) and \(x_j\) contributes a lot to the energy. In optimization problems, this effectively forces \(x_i\) and \(x_j\) to be close to each other.

In short:

> Large edge weight means strong coupling between the two vertices.

---

## Task 3: Kernel of the Laplacian

Let \(L\) be the Laplacian of an undirected graph.

We want to show that the constant vector

\[
\mathbf{1} = (1,1,\dots,1)^T
\]

satisfies

\[
L\mathbf{1}=0.
\]

---

### 1. Show that the constant vector satisfies \(L\mathbf{1}=0\)

Use the coordinate formula for the Laplacian:

\[
(Lx)_i = \sum_{j\sim i}(x_i-x_j).
\]

Now take \(x=\mathbf{1}\). Then for every vertex,

\[
x_i = 1
\quad \text{and} \quad
x_j = 1
\]

for all neighbors \(j\) of \(i\). Hence every term in the sum equals

\[
x_i-x_j = 1-1 = 0.
\]

Therefore

\[
(L\mathbf{1})_i = \sum_{j\sim i} 0 = 0
\]

for every vertex \(i\). So the whole vector is zero:

\[
L\mathbf{1}=0.
\]

This proves that the constant vector belongs to the kernel of the Laplacian.

---

### 2. Intuitive explanation

The Laplacian measures differences between a vertex and its neighbors. If the signal is constant on all vertices, then there are no differences anywhere in the graph.

So the Laplacian does nothing to a constant signal.

This is exactly why constant vectors lie in the kernel: they are perfectly smooth over the graph.

---

## Final summary

These tasks illustrate the most important ideas about the graph Laplacian:

1. **Locality**: applying \(L\) once uses only information from immediate neighbors.
2. **Propagation**: applying \(L\) repeatedly, as in \(L^k x\), lets information spread to vertices up to distance \(k\).
3. **Weights**: in weighted graphs, larger weights mean stronger interaction and stronger penalty for differences across an edge.
4. **Energy interpretation**: the quadratic form \(x^TLx\) measures how non-smooth the signal \(x\) is on the graph.
5. **Kernel**: constant signals are annihilated by the Laplacian, because they have no variation across edges.

These properties are fundamental in spectral graph theory and also explain why Laplacian-based operations are important in Graph Neural Networks.
