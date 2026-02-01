---
marp: true
theme: default
# class: invert
# paginate: true
# header: Kujath, Nils. 2025. *Topics in Machine Learning for Natural Language Processing*.
style: |-
  :root {
    font-size: 24px;
  }
  code, pre {
    color: black !important;
  }
  h1 {color: black;}
  h2 {
    position: absolute;
    top: 60px;
    left: 75px;
    right: 75px;
    margin-top: 10;
  }
  h3 {
    position: absolute;
    top: 100px;
    left: 75px;
    right: 75px;
    margin-top: 10;
  }
  section:not(:has(h1)) {
    justify-content: flex-start;
    padding-top: 140px;
  }
  pre, code {
    tab-size: 4;
  }
  img[alt~="center"] {
  display: block;
  margin: 0 auto;
  }
---
# Notes on Natural Language Processing

Nils Kujath

v2026.01

---
## From Sparse to Dense Word Representations

* Let $V = \{w_1, \ldots, w_{|V|}\}$ be a finite vocabulary. The canonical (standard) basis of $\mathbb{R}^{|V|}$ is the indexed set $\mathcal{B}_{\mathrm{can}} = \{\mathbf{e}_1, \ldots, \mathbf{e}_{|V|}\}$, where $\mathbf{e}_i$ denotes the $i$-th basis vector, i.e. the unique (one-hot) vector with a $1$ in coordinate $i$ and zeros elsewhere. The BoW construction  $\boxed{\iota : V \to \mathbb{R}^{|V|}, \quad w_i \mapsto \mathbf{e}_i}$ identifies each word $w_i \in V$ with the canonical basis vector $\mathbf{e}_i \in \mathbb{R}^{|V|}$.
* Under the standard inner product on $\mathbb{R}^{|V|}$, the BoW representation satisfies 
  $$ \mathbf{e}_u {}^\top \mathbf{e}_v = \delta_{uv} = \begin{cases} 1 & \text{if } u = v, \\ 0 & \text{if } u \neq v \end{cases}, \quad \forall u,v \in \{1,\ldots,|V|\}.$$
  Consequently, all distinct word types are mutually orthogonal (for any $u \neq v, \mathbf{e}_u {}^\top \mathbf{e}_v = 0$) and equidistant in $\mathbb{R}^{|V|}$ (for any $u \neq v$, $\|\mathbf{e}_u - \mathbf{e}_v\| = \sqrt{(1)^2 + (-1)^2} = \sqrt{2}$). That is, the induced geometry encodes no graded notion of similarity between words in $V$.
* To obtain representations with a non-degenerate geometry, the BoW map $\iota$ can be replaced by a learned embedding $\boxed{\mathrm{emb} : V \to \mathbb{R}^m, \quad w \mapsto \mathbf{v}_w, \quad m \ll |V|}$ , where word vectors are dense and the embedding space offers the possibility of expressing similarity between words as geometric proximity.
