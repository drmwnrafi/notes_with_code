<center><h2>Kalman Filter</h2></center>

Kalman FIlter berfungsi untuk melakukan prediksi dan estimasi pada *state* (keadaan) suatu sistem.

Kalman filter membutuhkan 2 model, yaitu state model dan observation model

1. State Model

   State model adalah persamaan / model yang merepresentasikan perubahan *state* pada sistem w.r.t waktu.

2. Observation Model

   Observation model adalah persaamaan / model yang merepresentasikan data yang diukur biasanya dengan sensor yang memberikan data eksternal robot dihubungkan dengan keadaan sistem.

### Multivariate Kalman Filter

State pada sistem$$
$$
\bold{x} = \begin{bmatrix}
x \\ y
\end{bmatrix}
$$
*Multivariate* Kalman Filter memberikan output berupa **multivariate random variabel** dan **covariance matrix** merupakan matriks persegi sebesar jumlah state ketidakpastian.

*Covariance matrix* pada **Kalman Filter** meliputi :

- $P_{n,n}$ : covariance matriks pada estimasi
- $P_{n+1,n}$ : covariance matriks pada prediksi
- $R_n$ : covariance matriks yang menggambarkan ketidakpastian pengukuran
- $Q$ : covariance matriks untuk proses noise

#### Covariance 

Covariance mengukur seberapa korelasi antara 2 atau lebih variabel random

![Object on the x-y plane](https://www.kalmanfilter.net/img/BB2/xyPlane.png)

Covaraince antara populasi $X$ dan populasi $Y$ dengan banyak data $N$:
$$
COV(X, Y) = \frac{1}{N}\sum_{i=1}^N(x_i - \mu_x)(y_i-\mu_y)
$$
Jika menggunakan sample:
$$
COV(X, Y) = \frac{1}{N-1}\sum_{i=1}^N(x_i - \mu_x)(y_i-\mu_y)
$$

#### Covariance Matrix

Coariance matrix adalah matriks persegi yang merepresentasikan covariance antar setiap element pada setiap state.
$$
\Sigma = \begin{bmatrix} 
\sigma_{xx} & \sigma_{xy} \\
\sigma_{yx} & \sigma_{yy}
\end{bmatrix}
= \begin{bmatrix} 
\sigma_{x}^2 & \sigma_{xy} \\
\sigma_{yx} & \sigma_{y}^2
\end{bmatrix}
= \begin{bmatrix} 
VAR(x)& COV(x,y) \\
COV(y,x) & VAR(y)
\end{bmatrix}
$$
Jika $x$ dan $y$ tidak berkolerasi maka bagian dari matriks $\Sigma$ yang bukan diagonal bernilai 0

Jika memiliki *state* $x$ dengan dimensi $1 \times k$ ,
$$
COV(x) = E((x-\mu_x)(x-\mu_x)^T)
$$
Proof :
$$
\begin{align*}
COV(x) &= E \left( \begin{bmatrix}
(x_1 -\mu_{x_1})^2 & (x_1-\mu_{x_1})(x_2-\mu_{x_2}) & \cdots &(x_1-\mu_{x_1})(x_k-\mu_{x_k})\\
(x_2-\mu_{x_2})(x_1 -\mu_{x1}) & (x_2-\mu_{x_2})^2 & \cdots &(x_2-\mu_{x_1})(x_k-\mu_{x_k})\\
\vdots & \vdots & \ddots & \vdots \\
(x_k-\mu_{x_k})(x_1 -\mu_{x1}) & (x_k-\mu_{x_k})(x_2-\mu_{x_2}) & \cdots &(x_k-\mu_{x_k})^2\\
\end{bmatrix} \right )\\
&= E \left( \begin{bmatrix}
(x_1 -\mu_{x_1})\\
(x_2-\mu_{x_2})\\
\vdots \\
(x_k-\mu_{x_k})
\end{bmatrix} 
\begin{bmatrix} [ (x_1 -\mu_{x_1})&(x_2-\mu_{x_2})&\cdots & (x_k-\mu_{x_k})
\end{bmatrix}
\right )\\
& = E((x_{1:k}-\mu_{x_{i:k}})(x_{1:k}-\mu_{x_{i:k}})^T)
\end{align*}
$$


Note :

$E(v)$ adalah ekspektasi, rata-rata dari variabel random

#### Covariance Matrix Properties

- $$
  \Sigma_{ii} = \sigma_{ii}^2
  $$

- $$
  tr(\Sigma) = \sum _{i=1}^n \Sigma_{ii}\geq 0
  $$

- $$
  \Sigma = \Sigma^T
  $$

### Multivariate Gaussian Distribution

$$
p(x|\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)|\Sigma|}}exp\left (-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right )
$$

### State Equation

Befungsi untuk melakukan prediksi berdasarkan estimasi yang sudah ada.

$\hat{x}_{n+1,n} = F\hat{x}_{n,n} + G\hat{u}_{n,n} +w_n$

$\hat{x}_{n+1,n}$ : prediksi state pada $n+1$

$\hat{x}_{n,n}$ : prediksi state pada $n$

$u_n$ : variabel kontrol *input* terukur ke dalam sistem

$w_n$ : noise, *input* tidak terukur ke dalam sistem (disebut proses noise)

$F$ : matrix transisi

$G$ : matrix kontrol (mapping efek $u_n$ ke variabel *state*)

Persamaan state dapat diperoleh dengan *State Space Equations* dengan menganlisa dynamic dari sistem.

### Covariance Equation

Mengukur ketidakpastian dari prediksi  *State Equation*

$P_{n+1,n} = FP_{n,n}F^T + Q$

Proof :

dengan $COV(x) = E((x_{1:k}-\mu_{x_{i:k}})(x_{1:k}-\mu_{x_{i:k}})^T)$, maka

$P_{n,n} = E((\hat{x}_{n,n}-\mu_{\hat{x}_{n,n}})(\hat{x}_{n,n}-\mu_{\hat{x}_{n,n}})^T)$ , dengan menggunakan *state equation* dapat diperoleh $P_{n+1, n}$
$$
\begin{align*}
P_{n+1,n} &= E((\hat{x}_{n+1,n}-\mu_{\hat{x}_{n+1,n}})(\hat{x}_{n+1,n}-\mu_{\hat{x}_{+1,n}})^T)\\
&= E \left ( (F\hat{x}_{n,n} + G\hat{u}_{n,n} - F\hat{\mu_x}_{n,n} + G\hat{u}_{n,n}) (F\hat{x}_{n,n} + G\hat{u}_{n,n} - F\hat{\mu_x}_{n,n} + G\hat{u}_{n,n})^T \right )\\
&=E \left ( F(\hat{x}_{n,n} - \mu_{x_{n,n}}) (F(\hat{x}_{n,n} - \mu_{x_{n,n}}))^T\right)\\
&=E \left ( F(\hat{x}_{n,n} - \mu_{x_{n,n}}) (\hat{x}_{n,n} - \mu_{x_{n,n}})^T F^T\right)\\
&=FE \left (\hat{x}_{n,n} - \mu_{x_{n,n}}) (\hat{x}_{n,n} - \mu_{x_{n,n}})^T\right)F^T\\
\text{sehingga,}\\
P_{n+1,n} &= FP_{n,n}F^T
\end{align*}
$$

#### Observation Model

$z_n = Hx_n+v_n$

$z_n$ : vektor observasi

$H$ : matriks observasi

$x_n$ : state sebenarnya

$v_n$ : random noise (disebut noise observasi)

Contoh pada kasus sensor jarak ultrasonik, state dari sistem adalah jarak $x_n$, output dari sensor merupakan ToF yang terukur $z_n$, maka *obeservation model*-nya :

$z_n = [\frac{2}{c}]x_n + v_n$

dengan $c$ merupakan kecepatan suara

#### Observation Uncertainty

$R_n = E(v_nv_n^T)$

$R_n$ : covariance matrix untuk obervasi

$v_n$ : kesalahan observasi

#### Process Uncertainty

$Q_n = E(w_nw_n^T)$

$Q_n$ : covariance matrix untuk proses noise

$w_n$ : proses noise

#### Summary

![The Kalman Filter Diagram](https://www.kalmanfilter.net/img/summary/KalmanFilterDiagram.png)

#### Kalman Gain

Kalman Gain merupakan bobot bagi *measurement model*, dengan kata lain *Kalman Gain* menentukan seberapa besar pengaruh *measurement model* terhadap estimasi *state*.
$$
KG = \frac{error_{estimate}}{error_{estimate} + error_{measurement}}
$$

- $error_{estimate} > error_{measurement} $ :  $KG \rightarrow 1$
- $error_{estimate} < error_{measurement} $ :  $KG \rightarrow 0$

$$
estimate_t = estimate_{t-1} + KG(measurement_t-estimate_{t-1})
$$

#### Error on Estimation

$$
E_{estimate_t} = [1-KG]E_{estimate_{t-1}}
$$

- $KG\rightarrow1$, maka  error pada estimasinya besar, sehingga persamaan diatas akan menghasilkan $E_{estimate}$ yang kecil 
- $KG\rightarrow0$, maka  error pada estimasinya kecil, sehingga persamaan diatas akan mempertahankan error estimasi dari waktu $t-1$ 
- $E_{estimate_t} < E_{estimate_{t-1}}$  

Note :

Saat varians pada estimasi mengecil, maka mendekati nilai sebenarnya

#### References :

- [kalmanfilter.com](https://www.kalmanfilter.net/)
- [Michel van Biezen - Kalman Filter Playlist](https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT)