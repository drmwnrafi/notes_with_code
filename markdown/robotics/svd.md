<center><h1>Singular Value Decomposition (SVD)</h1></center>

Semua matriks dapat diuraikan menjadi 3 matriks baru,
$$
SVD(A) = U\Sigma U^T\\
A = \text{input matrix}\\
U = \text{orthogonal matrix (m x m), left singular values} \\
\Sigma = \text{diagonal singular values matrix (m x n)} \\
V = \text{orthogonal matrix (n x n), right singular values}
$$
![image-20240617133616487](/home/ziczac/snap/typora/88/.config/Typora/typora-user-images/image-20240617133616487.png)

### Orthogonal Matrix

Matrix yang kolom dan barisnya merupakan vektor orthonormals
$$
Q^TQ = QQ^T = I, \text{where } Q^T = Q^{-1}
$$
Suatu vektor dapat dikatakan orthonormal apabila memiliki magnitude = 1 dan dot product ($\cdot$) = 0.

Contoh pada matriks rotasi :
$$
R = \begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
$$
 Matriks rotasi merupakan orthogonal karena $RR^T = I$, dan $R^T = R^{-1}$

### Singular Values

Singular values pada dasarnya merupakan akar dari eigenvalues ($\lambda_1, \lambda_2, \lambda_3, ..., \lambda_n$) dari $A^TA$, maka singular valuesnya adalah $\sigma_1 = \sqrt\lambda_1, \sigma_2 = \sqrt\lambda_2, \sigma_3 = \sqrt\lambda_3, ..., \sigma_n = \sqrt\lambda_n $. Matriks $A^TA$ merupakan matriks symmetric $n\times n$.

Matriks diagonal $\Sigma$, memiliki urutan $\sigma_n$ dari paling besar ke paling kecil dan $\geq 0$​.

### Rank Matrix

Rank matriks merupakan nilai kolom atau baris yang independen secara linear.

Contoh :
$$
A = \begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9\\
\end{bmatrix}, R_2=R_2-4R_1\\
A = \begin{bmatrix}
1&2&3\\
0&-3&-6\\
7&8&9\\
\end{bmatrix}, R_3 = R_3-7R_1\\
A = \begin{bmatrix}
1&2&3\\
0&-3&-6\\
0&-6&-12\\
\end{bmatrix}, R_2 = \frac{R_2}{R_3}\\
A = \begin{bmatrix}
1&2&3\\
0&1&2\\
0&-6&-12\\
\end{bmatrix}, R_3=R_3+6R3\\
A = \begin{bmatrix}
1&2&3\\
0&1&2\\
0&0&0\\
\end{bmatrix}
$$
Maka, jumlah rank dari matriks $A$ adalah 2 (non-zeros baris atau kolom). 

### How To 

1. Hitung $A^TA$ adalah matriks simetrik, maka eigenvectornya orthogonal. Untuk mendapatkan matriks $V^T$
2. Hitung eigenvalues ($\lambda$) dengan $\text{det}(A^TA-\lambda I)=0$, maka diperoleh eigenvalues $\lambda_1,\lambda_2, .., \lambda_n$
3. Hitung eigenvector ($v$) dengan menggunakan $\lambda_{1:n}$. Agar magnitude eigenvector = 1, tiap element pada eigenvector dibagi dengan $\sqrt{v_i^2+v_j^2+...+v_n^2}$ 
4. Hitung nilai singular value $\sigma_i = \sqrt{\lambda_i}$​
5. Hitung langkah 1 sampai 3, dengan $AA^T$ untuk mendapatkan matriks $U$