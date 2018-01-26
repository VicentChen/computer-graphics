# Chapter 7

1. $$\begin{align}
&\exists z_1>z_2, (x_1,y_1,z_1,1), (x_2,y_2,z_2,1)\\
&\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&\frac{n+f}{n}&-f\\0&0&\frac{1}{n}&0\end{bmatrix}\begin{bmatrix}x\\y\\z\\1\end{bmatrix}=\begin{bmatrix}x\\y\\\frac{z}{n}(n+f)-f\\\frac{z}{n}\end{bmatrix}\\
\frac{n+f}{n}=Constant C,z_1>z_2, \frac{z_1}{n}(n+f)-f>\frac{z_2}{n}(n+f)-f
\end{align}$$

2. $$\begin{align}
X\begin{bmatrix}hx\\hy\\hz\\h\end{bmatrix}=hX\begin{bmatrix}x\\y\\z\\1\end{bmatrix}=h\begin{bmatrix}x'\\y'\\z'\\1'\end{bmatrix}=\begin{bmatrix}x'\\y'\\z'\\1'\end{bmatrix}
\end{align}$$

3. $$\begin{align}
M_pM_p^{-1}=\begin{bmatrix}n&0&0&0\\0&n&0&0\\0&0&n+f&-fn\\0&0&-1&0\end{bmatrix}\begin{bmatrix}\frac{1}{n}&0&0&0\\0&\frac{1}{n}&0&0\\0&0&0&1\\0&0&-\frac{1}{fn}&\frac{n+f}{fn}\end{bmatrix}=E
\end{align}$$

4. $$\begin{align}
M_{projection}(r,t,n)&=\begin{bmatrix}\frac{2n}{r-l}&0&\frac{l+r}{l-r}&0\\0&\frac{2n}{t-b}&\frac{b+t}{b-t}&0\\0&0&\frac{f+n}{n-f}&\frac{2fn}{f-n}\\0&0&1&0\end{bmatrix}\begin{bmatrix}r\\t\\n\\1\end{bmatrix}\\
&=\begin{bmatrix}\frac{2nr-nl-nr}{r-l}\\\frac{2nt-nb-nt}{t-b}\\\frac{nf+n^2-2fn}{n-f}\\n\end{bmatrix}\\
&=\begin{bmatrix}1\\1\\1\\1\end{bmatrix}
\end{align}$$

5. $$\begin{align}
\begin{bmatrix}\frac{2}{r-l}&0&\frac{l+r}{l-r}&0\\0&\frac{2}{t-b}&\frac{b+t}{b-t}&0\\0&0&-3&4\\0&0&1&0\end{bmatrix}
\end{align}$$

6. $$\begin{align}
M_pP=\begin{bmatrix}n&0&0&0\\0&n&0&0\\0&0&n+f&-fn\\0&0&-1&0\end{bmatrix}\begin{bmatrix}x\\y\\z\\1\end{bmatrix}=\begin{bmatrix}nx\\ny\\nz+fz-fn\\-z\end{bmatrix}=\begin{bmatrix}-\frac{nx}{z}\\-\frac{ny}{z}\\\frac{fn}{z}-n-f\\1\end{bmatrix}
\end{align}$$
