<center><h2>Singular Value Decomposition - Iterative Closest Point (SVD-ICP)</h2>
</center>



$$
P = \left \{ p_1, p_2, p_3, ..., p_n \right \}\\
Q = \left \{ q_1, q_2,q_3, ..., q_n \right \}\\
E = \underset{R,t} {\mathrm{argmin}} \sum_{i=1}^n \left||Rp_i + t -q_i \right||^2
$$
$E$ adalah fungsi loss, $P$ adalah titik korespondensi pada *pointcould* source dan $Q$ adalah titik korespondensi pada pointcloud reference. Tujuan ICP adalah menemukan transformasi terbaik yang dapat menayelarasakn *pointcloud* $P$ ke $Q$.
$$
\mu_p = \frac{1}{N} \sum_{n=1}^N q_i\\
\mu_q = \frac{1}{N} \sum_{n=1}^N p_i
$$
$\mu_q$ dan $\mu_p$ merupakan center of mass dari pointcloud $Q$ dan $P$. Kemudian, tentukan koordinat lokal pada pointcloud $Q$ dan $P$ dengan mengurangkan tiap titik $Q$ dan $P$ terhadap center of mass-nya.
$$
Q' = {Q_i-\mu_q}=\{Q'_i\}\\
P' = {P_i-\mu_p}=\{P'_i\}\\
W = \sum_{i=1}^N q'_ip'^T_i\\
W = \begin{bmatrix}
cov(p_x,q_x) & cov(p_x,q_y)\\
cov(p_y,q_x) & cov(p_y,q_y)
\end{bmatrix}
$$
$W$ merupakan cross-covariance, cross-covariance memberikan informasi bagaimana perubahan pada koordinat di titik $p$ jika koordinat titik $x$ berubah. Cross-covariance yang ideal adalah matriks identitas, dikarenakan pada kondisi ideal tidak ada korelasi antara axis-$x$ dan axis-$y$. 
$$
R = VU^T\\
t = \mu_y-R\mu_x
$$

### Vektor Translasi $\left ( t = \mu_p-R\mu_x\right)$

Turunkan $E$ terhadap $t$,
$$
\frac{\partial E}{\partial t} = 2\sum_{i=1}^n (y_i - Rx_i-t) = 0\\
t = y_i-Rx_i
$$


### Matriks Rotasi $\left ( R = VU^T\right )$

Rotasi $R$ mencapai optimal jika translasi $t= 0$,
$$
R= \underset{R\in SO(n)}{argmin}\sum_{i=1}^n\left\|Rx_i - y_i \right\|^2
\\ 
\begin{align*}
\left\|Rx_i - y_i \right\|^2 &= (Rx_i - y_i)^T(Rx_i - y_i) \\
&= x_i^T R^T R x_i - y_i^T R x_i - x_i^T R^T y_i - y_i^T y_i \\
&= x_i^T x_i - y_i^T R x_i - x_i^T R^T y_i - y_i^T y_i
\end{align*}
$$
$x_i^TR^Ty_i$ adalah operasi skalar, sehingga $a=a^T$
$$
x_i^TR^Ty_i = \left ( x_i^TR^Ty_i\right)^T = y_i^TRx_i\\
\underset{R\in SO(d)}{\mathrm{argmin}}\sum_{i=1}^n \left\|Rx_i - y_i \right\|^2 = 
\underset{R\in SO(d)}{\mathrm{argmin}} \sum _{i=n}^n x_i^T x_i - 2y_i^T R x_i - y_i^T y_i
$$
untuk mendapatkan matriks rotasi $R$, persamaan $x_i^Tx_i$ dan $y_i^Ty_i$ dapat diabaikan. Sehingga, untuk meminimalkan nilai $\chi^2$, sama dengan memasimalkan nilai :
$$
\underset{R\in SO(d)}{\mathrm{argmax}}\sum_{i=1}^ny_i^T R x_i\\
\sum_{i=1}^ny_i^T R x_i = \text{tr}(Y^TRX) = \text{tr}(RXY^T) =\text{tr}(RW)
$$
dengan sifat trace pada matrix $\text{tr}(AB) = \text{tr}(BA)$ dan $XY^T = W$, sehingga
$$
SVD(W) = U \Sigma V^T, \text{subtitute to } tr(RXY^T) \\
\text{tr}(RU\Sigma V^T) = \text{tr}(\Sigma V^TRU)
$$
tetapkan matriks $M =V^TRU$, $M$ merupakan matriks orthogonal. Sehingga,
$$
\text{tr}(\Sigma V^TRU) = \text{tr}(\Sigma M)
$$
dengan menerapkan Cauchy-Schwarz inequality pada $\text{tr}(\Sigma M)$,
$$
\text{tr}(\Sigma M)= \sum_{i=1}^r\sigma_im_{ii}\\
\left(\sum_{i=1}^r\sigma_im_{ii}\right)^2 \leq \sum_{i=1}^r\sigma_i^2  \sum_{i=1}^r m_{ii}^2\\
\sum_{i=1}^r\sigma_im_{ii} \leq \sqrt{\sum_{i=1}^r\sigma_i^2  \sum_{i=1}^r m_{ii}^2}\\
\sum_{i=1}^r\sigma_im_{ii} \leq \sum_{i=1}^r\sigma_i^2
$$
dikarenakan $M$ merupakan matriks orthogonal, maka $m_{ii}^2 = |m_{ii}|\leq1$. Sehingga, untuk memperoleh nilai maksimum $m_{ii}$ bernilai 1. Seperti yang dijelaskan diawal bahwa $M$ merupakan matriks orthogonal, sehinggga $m_{ii}$ bernilai 1 jika $M = I$​
$$
M = I = V^TRU\\
R^T = V^TU\\
R = UV^T
$$

### References :

- Sorkine-Hornung, O., & Rabinovich, M. (2017). Least-squares rigid motion using svd. *Computing*, *1*(1), 1-5. https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
- Arun, K. S., Huang, T. S., & Blostein, S. D. (1987). Least-Squares fitting of two 3-D point sets. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *PAMI-9*(5), 698–700. https://doi.org/10.1109/tpami.1987.4767965