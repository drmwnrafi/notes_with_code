<center> <h2>Levenberg Marquardt - Point to Line Iterative Closest Point (LM-PLICP)</h2></center>

### Gradient-Descent

$$
x_{i+1} = x_i + \lambda\nabla_{f(x)}
$$

### Newton-Rapshon

$$
x_{i+1} = x_i + (\nabla^2_{f(x_i)})^{-1}\nabla_{f(x)}
$$

### Gauss-Newton

$$
x_{i+1} = x_i + (J_{f(x)}^TJ_{f(x)})^{-1}\nabla_{f(x)}
$$

### Levenberg-Marquardt

$$
x_{i+1} = x_i + (\nabla^2_{f(x_i)}+\lambda \; \text{diag}(\nabla^2_{f(x_i)}))^{-1}\nabla_{f(x)}
$$

### Point to Line Iterative Closest Point

$$
E = \underset{R,t} {\mathrm{argmin}} \sum_{i=1}^n \left||(y_i - Rx_i + t)n_{y,i} \right||^2
$$

$n_{y,i}$ vektor normal pada point cloud reference dihitung dengan $\frac{-dy}{dx}$

### Approximation Hessian $H=J^TJ$

Fungsi loss $E$ merupakan fungsi non-linear *differentiable* dan smooth dikarenakan terdapat matriks rotasi $R$,  sehingga Taylor-Expansion valid untuk menyelesaikan fungsi non-linear $E$
$$
\text{Taylor-Expansion :}\\
F(x+\Delta x)=F(x)+\frac{1}{1!}F'(x)\Delta x+\frac{1}{2!}F''(x)\Delta x^2+...
$$
Sehingga,
$$
y(x)=f(x; s)\\
r_i = y_i - f(x_i; s)\\
$$
$f(x; s)$ merupakan fungsi non-linear, $r_i$ merupakan residual, $y_i$ merupakan target, dan $s$ merupakan state yaitu $( t_x, t_y, \theta)$ yang akan dioptimisasi, dengan **First Order Taylor-Expansion** untuk menghitung perubahan pada state.
$$
f(x_i; s') = \underbrace{f(x_i; s)}_{y_i-r_i} + \frac{\partial f}{\partial s}(s'-s)=y_i
$$
Agar $f(x_1; s') = y_i$ maka $\frac{\partial f}{\partial s}(s'-s) = r_i$, ubah $s$ menjadi $( t_x, t_y, \theta)$,
$$
\frac{\partial f}{\partial t_x}\Delta t_x+\frac{\partial f}{\partial t_y}\Delta t_y+\frac{\partial f}{\partial \theta}\Delta \theta =r_1\\
\frac{\partial f}{\partial t_x}\Delta t_x+\frac{\partial f}{\partial t_y}\Delta t_y+\frac{\partial f}{\partial \theta}\Delta \theta =r_2\\
\;\;\;\;\;\;\vdots\;\;\;\;\;+\;\;\;\;\;\;\vdots \;\;\;\;\;\;\;+ \;\;\;\;\;\vdots\;\;\;=\;\vdots\\
\frac{\partial f}{\partial t_x}\Delta t_x+\frac{\partial f}{\partial t_y}\Delta t_y+\frac{\partial f}{\partial \theta}\Delta \theta =r_n
$$
Sehingga dapat disederhanakan menjadi, 
$$
\begin{matrix}
x=x_1\\
\vdots\\
x=x_n
\end{matrix}
\begin{bmatrix}
\frac{\partial f}{\partial t_x} & \frac{\partial f}{\partial t_y}& \frac{\partial f}{\partial \theta}\\
\vdots&\vdots&\vdots\\
\frac{\partial f}{\partial t_x} & \frac{\partial f}{\partial t_y}& \frac{\partial f}{\partial \theta}\\
\end{bmatrix}
\begin{bmatrix}
\Delta t_x\\
\Delta t_y\\
\Delta \theta
\end{bmatrix} =
\begin{bmatrix}
r_1\\
\vdots\\
r_n\\
\end{bmatrix}
$$
Sehingga,
$$
J \Delta s=r
$$
Dikarenakan matriks $J$ bukan matriks persegi, sehingga matriks $J$ ***non-invertible*** matriks. Agar dapat menyelesaikan persamaan tersebut, kedua sisi dikalikan dengan $J^T$, menjadi
$$
(J^TJ)\Delta s=J^Tr\\
\Delta s = (J^TJ)^{-1}J^Tr
$$

#### References :

- [METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS](https://www.researchgate.net/publication/261652064_Methods_for_Non-Linear_Least_Squares_Problems_2nd_ed)
- [Nonlinear Least Squares](https://www.youtube.com/watch?v=8evmj2L-iCY)