# Chapter 5

1. `to do`
2. $A$的行标准正交
$$
\begin{align}
&A=\begin{pmatrix}a_{11} &\cdots &a_{1n}\\\vdots&\ddots&\vdots\\ a_{m1}&\cdots&a_{mn}\end{pmatrix}\\
&AA^T=\begin{pmatrix}1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&1\end{pmatrix}\\
&A^TA=\begin{pmatrix}1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&1\end{pmatrix}\\
&\Sigma_{i=0}^n a_{ij}=1, \Sigma_{i=0}^ma_{ij}a_{ik}=0
\end{align}
$$
$A$的列标准正交

3. $$
\begin{align}
&A=\begin{bmatrix}a_{11}&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&a_{nn}\end{bmatrix}\\
&|A-\lambda I|=\begin{vmatrix}a_{11}-\lambda_1&\cdots&0\\\vdots&\ddots&\vdots\\0&\cdots&a_{nn}-\lambda_n\end{vmatrix}=0\\
&\Rightarrow(a_{11}-\lambda_1)\cdots(a_{nn}-\lambda_n)=0\\
&\Rightarrow\lambda = (a_{11},\cdots,a_{nn})
\end{align}
$$

4. $$
\begin{align}
(AA^T)^T=(A^T)^TA^T=AA^T
\end{align}
$$

5. $$
\begin{align}
&a=(a_1,a_2,a_3),b=(b_1,b_2,b_3),c=(c_1,c_2,c_3)\\
&\begin{vmatrix}a_1&b_1&c_1\\a_2&b_2&c_2\\a_3&b_3&c_3\end{vmatrix}=a_1b_2c_3+a_2b_3c_1+a_3b_1c_2-a_3b_2c_1-a_2b_1c_3-a_1b_3c_2\\
&(a\times b)\cdot c=\begin{bmatrix}a_2b_3-b_2a_3\\a_3b_1-b_3a_1\\a_1b_2-b_1a_2\end{bmatrix}\cdot c=a_1b_2c_3+a_2b_3c_1+a_3b_1c_2-a_3b_2c_1-a_2b_1c_3-a_1b_3c_2\\
&|abc|=(a\times b)\cdot c
\end{align}
$$

6. $\frac{1}{6}|abc|=\frac{1}{6}(a\times b)\cdot c$
$S_{bottom}=|a\times b|, H=|c|\cdot cos\theta$
$\theta$为c与底面法向量夹角
$V_{cube}=SH=|a\times b||c|cos\theta=(a\times b)\cdot c$
$V_{tetrahedron}=\frac{1}{6}SH=\frac{1}{6}(a\times b)\cdot c$

7. $$
\begin{align}
&\frac{1}{2}\begin{vmatrix}x_0&x_1&x_2\\y_0&y_1&y_2\\1&1&1\end{vmatrix}=\frac{1}{2}\begin{vmatrix}1&1&1\\x_0&x_1&x_2\\y_0&y_1&y_2\end{vmatrix}\\
&\Rightarrow\frac{1}{2}[(x_1y_2-x_2y_1)-(x_0y_2-x_2y_0)+(x_0y_1-x_1y_0)]\\
&\Rightarrow\frac{1}{2}[S_{square}-S_{square}+S_{square}]\\
&\Rightarrow\frac{1}{2}S_{square}=S_{triangle}\\
\end{align}
$$