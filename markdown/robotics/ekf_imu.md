<h1><center>Extended Kalman Filter</center></h1>

## Hamilton's Quternions

Konsep Quaternion diperkenalkan oleh **Sir William Rowan Hamilton** (thanks to this genius man), 
$$
i^2=j^2=k^2=ijk=-1
$$

 ### Complex Number

Bilangan kompleks (*complex numbers*) merupakan konsep yang menggunakan bilangan imajiner. Imajiner digunakan untuk menyelesaikan persamaan yang tidak memiliki solusi.
$$
x^2+1=0 \; \text{maka, }\\
x^2=-1
$$
Persamaan tersebut tidak memiliki solusi, karena bilangan rill $\R$ apabila dipangkat kuadrat akan bernilai $+$. Sehingga ditetapkan $i^2 =-1$ (don't tell me to explain math proof of this, just accept it). 

Himpunan dari bilangan kompleks dinotasikan dengan $\mathbb{C}$ merupakan hasil jumlah bilangan rill dengan imajiner, 
$$
z=a+bi \; \; \; a,b \in \R, \; \;i^2=-1
$$
 Sifat Complex :

- $(a_1+b_1i)+(a_2+b_2i)=(a_1+a_2)+(b_1+b_2)i$
- $(a_1+b_1i)-(a_2+b_2i)=(a_1-a_2)+(b_1-b_2)i$
- $c(a+bi)=ca+cbi, \text{ c = konstanta}$
- $z_1z_2=(a_1a_2−b_1𝑏_2)+(a_1𝑏_2+b_1a_2)𝑖$
- $z^*=a-bi$
- $|z|=\sqrt{zz^*}=\sqrt{a^2+b^2}$

Bilangan kompleks pada trigonometri di representasikan dengan $z = \cos(\theta)+ i \sin(\theta)$, dan pada representasi Euler $e^{i\theta} = \cos(\theta)+ i \sin(\theta)$.

### Quaternions

Quaternion dinotasikan dengan $\H$ (Hamilton), merupakan bilangan "kompleks" pada 3D-sphere  $S^3$ dengan menambahkan 2 bilangan imajiner , $j$ dan $k$.  Sehinga direpresentasikan dengan persamaan,
$$
q = w+xi+yj+zk,\;\;\;w,x,y,z\in\R
$$
dengan $i^2=j^2=k^2=ijk=-1$, dan
$$
ij = k \;\;\;jk=i\;\;\;ki=j\\
ji=-k\;\;\;kj=-i\;\;\;ik=-j
$$
Quaternion dapat direpresentasikan dengan,
$$
q=[w, \textbf{v}] \;\;\;\;w\in\R, \textbf{v}\in\R^3\\
\textbf{v}=xi+yj+zk\;\;\;\;\;\;\;w,x,y,z\in\R
$$
Sifat Quaternion:

- $q_1+q_2=[w_1+w_2, \textbf{v}_1+\textbf{v}_2]$

- $q_1-q_2=[w_1-w_1, \textbf{v}_1-\textbf{v}_2]$

- $$
  \begin{flalign}
  q_1 =& [w_1, \textbf{v}_1]\\
  q_2 =& [w_2, \textbf{v}_2]\\
  q_1 q_2 = & \ (w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2) \\
            & + (w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2)i \\
            & + (w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2)j \\
            & + (w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2)k \\
           =& \left[ w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2, w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2 \right]
  \end{flalign}
  $$

  

- $\text{c}q=[\text{c}w, \text{c}\textbf{v}], \text{ c = konstanta}$

- $$
  q_1 = [0, \textbf{v}_1]\;\;\;q_2 = [0, \textbf{v}_2]\\
  q_1q_2 = [-\textbf{v}_1\cdot\textbf{v}_2, \textbf{v}_1\times\textbf{v}_2]
  $$

Representasi quaternion pada rotasi titik di 3D,
$$
q = [\cos(\theta), \sin(\theta)\textbf{v}]
$$


##### Note:

- **Cross Product**  

  $\vec{a}\times\vec{b}=|\vec{a}| \left|\vec{b}\right|\sin(\theta)\vec{n}$

  ![cross product with angle and unit vector](https://www.mathsisfun.com/algebra/images/cross-product.svg)

- **Dot Product** 

  $\vec{a}\cdot\vec{b}=|\vec{a}| \left|\vec{b}\right|\cos(\theta)$

  ![dot product magnitudes and angle](https://www.mathsisfun.com/algebra/images/dot-product-1.svg)

## Quaternion for 3D Rotation

## Gyroscope



## Accelerometer

## Magnetometer

##### References

- [Understanding Quaternions](https://www.3dgep.com/understanding-quaternions/#Introductio
