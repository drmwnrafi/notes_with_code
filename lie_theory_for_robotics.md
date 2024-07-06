<h1><center>Scratch Notes of Lie Theory for Robotics</center></h1>

## Group Theory

$\mathbb{K} = \left \{\R, \C, \H \right \}$

$\R$ adalah real number

$\C$ adalah complex number

$\H$ adalah quaternion number

### General Linear Group

Matrix yang dapat dilakukan operasi inverse dan determinant $\neq 0$.

$\text{GL}_n(\mathbb{K}) =\left\{ A \in M_n(\mathbb{K}): \det(A) \neq 0\right \}$

#### Theorem

- $\text{GL}_n(\mathbb{\C})$ isomorphic dengan subgroup $\text{GL}_{2n}(\mathbb{R})$

- $\text{GL}_n(\mathbb{\H})$ isomorphic dengan subgroup $\text{GL}_{2n}(\mathbb{C})$

Sehingga, $\text{GL}_n(\mathbb{\H})$ isomorphic dengan subgroup $\text{GL}_{4n}(\mathbb{R})$

##### Notes :

Jika group $G$ isomorph terhadap group $V$, maka kedua group tersebut memiliki struktur yang sama meskipun element-elementnya memiliki kemungkinan berbeda. Dengan catatan terdapat operasi untuk memetakan group $G$ ke group $V$, $f : G \rightarrow V$, dan sebaliknya.

### Affine Groups

##### References :

- [Matrix Lie Groups for Robotics - Winter 2020](https://www.youtube.com/watch?v=xIrtF2ACBrc&list=PLdMorpQLjeXmbFaVku4JdjmQByHHqTd1F&index=22)