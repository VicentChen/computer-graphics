# CMU-15462-Assigment_0.0
 - Origin Andrew ID: ab1
 - Date: August 30, 2017
 - **Correctness not guaranteed**

# 1 Linear Algebra
## 1.1 Basic Vector Operations
**Exercise 1**
$$
\begin{align}
(a)&u+v=(8,4)+(5,5)=(13,9)\\
(b)&bu=8*(8,4)=(64,32)\\
(c)&au-bv=4*(8,4)-8*(5,5)=(-8,-24)
\end{align}
$$
**Exercise 2**
$$
\begin{align}
(a)&u-v=(8,3,8)-(7,4,7)=(1,-1,1)\\
(b)&u+7v=(8,3,8)-7*(7,4,7)=(-41,-25,-41)
\end{align}
$$
**Exercise 3**
$$
\begin{align}
(a)&p(x)-q(x)=8x^2+3x+8-(7x^2+4x+7)=x^2-x+1\\
(b)&p(x)+7q(x)=8x^2+3x+8+7*(7x^2+4x+7)=57x^2+31x+57
\end{align}
$$

## 1.2 Inner Products and Norms
**Exercise 4**
$$u\cdot v=(8,2,8)\cdot(6,3,4)=8*6+2*3+8*4=48+6+32=86$$
**Exercise 5**
$$|u|=\sqrt{u\cdot u}=\sqrt{3^2+7^2+5^2}=\sqrt{83}$$
**Exercise 6**
$$
\begin{align}
(a)&\langle x,x\rangle=5*1*1+1*0+0*1+5*0*0=5\\
(b)&\langle y,y\rangle=5*0*0+0*1+1*0+5*1*1=5\\
(c)& \langle u,v\rangle- \langle v,u\rangle=5*2*7+2*3+3*7+5*3*3-(5*7*2+7*3+2*3+5*3*3)=0\\
(d)&\langle 6u+v,w\rangle-(6langle u,w\rangle+langle v,w\rangle)\\
&=\langle (18,41),(2,6)\rangle-(6\langle (2,6),(2,6)\rangle+\langle (6,5),(2,6)\rangle)
&=5*18*2+18*6+41*2+5*41*6-(6*(5*2*2+2*6+6*2+5*6*6)+(5*6*2+6*6+5*2+5*5*6))
&=1600-1600\\
&=0
\end{align}
$$
**Exercise 7**
$$
\begin{align}
(a)\langle \langle f,f\rangle\rangle&=\int_0^1(ax^2+b)^2dx\\
&=\int_0^1a^2x^4+2abx^2+b^2dx\\
&=[\frac{a^2}{5}x^5+\frac{2ab}{3}x^3+\frac{1}{3}b^3]_0^1\\
&=\frac{a^2}{5}+\frac{2ab}{3}\\
&=\frac{5}{16}+\frac{56}{3}\\
(b)\langle \langle f,f\rangle\rangle&=\int_0^1(6e^{7x})^2dx\\
&=\int_0^1(36e^{14x})dx\\
&=[\frac{18}{7}e^{14x}]_0^1\\
&=\frac{18}{7}e^{14}-\frac{18}{7}\\
(c)\langle \langle f,g\rangle\rangle&=\int_0^1(4x+2)*4x^2dx\\
&=\int_0^1(16x^3+8x^2)dx\\
&=[4x^4+\frac{8}{3}x^3]_0^1\\
&=16+\frac{8}{3}
\end{align}
$$
**Exercise 8**
$$
\begin{align}
||f||&=\sqrt{\langle\langle f,f\rangle\rangle}\\
&=\sqrt{\int_0^1(64e^{6x})dx}\\
&=\sqrt{[\frac{32}{3}e^{6x}]_0^1}\\
&=\sqrt{\frac{32}{3}e^{6}-\frac{32}{3}}
\end{align}
$$

## 1.3 Linear Maps
**Exercise 9**
$$
\begin{align}
(a)&f(x+y)-(f(x)+f(y))\\
&=3(x+y)+9-(3x+9+3y+9)\\
&=-9\\
(b)&f(5x)-5f(x)\\
&=(3*5x+9)-5*(3x+9)\\
&=-36\\
(c)&g(x+y)-(g(x)+g(y))\\
&=6(x+y)-(6x+6y)\\
&=0\\
(d)&g(7x)-7g(x)\\
&=6*7x-7*6x\\
&=0\\
(e)&h(x+y)-(h(x)+h(y))\\
&=-(x+y)^2-(-x^2-y^2)\\
&=-2xy\\
&=-90\\
(f)&h(5x)-5h(x)\\
&=-(5x)^2-5*(-x^2)\\
&=-20x^2
\end{align}
$$
**Exercise 10**
$$
\begin{align}
(a)&f(w_1u+w_2u)-(w_1f(u)+w_2f(v))\\
&=w_1u_1+w_2u_1+w_1u_2+w_2u_2+9-(w_1u_1+w_1u_2+w_2u_1+w_2u_2+9)\\
&=0\\
(b)&g(w_1u+w_2v)-(w_1g(u)+w_2g(v))\\
&=(w_1u_1+w_2v_1)^2+(w_1u_2+w_2v_2)^2-(w_1u_1^2+w_1u_2^2+w_2v_1^2+w_2v_2^2)\\
&=8.6^2+4.4^2-(0.4*64+0.4*64+0.6*81+0.6*4)
\end{align}
$$

**Exercise 11**
$$
\begin{align}
(a)&F(f+g)-(F(f)+F(g))\\
&=F(sin(x)+e^x)-(F(sinx(x))+F(e^x))\\
&=cos(x)+e^x+9-(cos(x)+9+e^x+9)\\
&=-9\\
(b)&F(4f)-4F(f)\\
&=4cos(x)+9-4*(cos(x)+9)\\
&=-27\\
(c)&G(f+g)-(G(f)+G(g))\\
&=\int_0^1sin(x)+e^xdx-(\int_0^1sin(x)dx+\int_0^1e^xdx)\\
&=[-cos(x)+e^x]_0^1-([-cos(x)]_0^1+[e^x]_0^1)\\
&=0\\
(d)&G(5f)-5G(f)\\
&=\int_0^15sin(x)dx-5\int_0^1sin(x)dx\\
&=[-5cos(x)]_0^1-5[-cos(x)]_0^1\\
&=0\\
(e)&H(f+g)-(H(f)+H(g))\\
&=sin(0)+e^0-(sin(0)+e^0)\\
&=0\\
(f)&H(6f)-6H(f)\\
&=6sin(0)-6*sin(0)\\
&=0\\
\end{align}
$$

## 1.4 Basis and Span
**Exercise 12**
$$
\begin{align}
(a)&a=7\sqrt2\\
(b)&b=2\sqrt2\\
(c)&u-(ae_1+be_2)=(4,0)
\end{align}
$$

**Exercise 13**
$$
u\cdot e_1e_1+u\cdot e_2e_2=(20,13)
$$

**Exercise 14**
$$
\begin{align}
(a)&\widetilde e_1=e_1/|e_1|=(\frac{1}{\sqrt5},\frac{2}{\sqrt5})\\
(b)&\hat e_2=e_2-\langle e_2,\widetilde e_1\rangle\widetilde e_1=(9,7)-(\frac{23}{5},\frac{46}{5})=(\frac{22}{5},-\frac{11}{5})\\
(c)&\widetilde e_2=\hat e_2/|\hat e_2|=\frac{11}{\sqrt5}(\frac{22}{5},-\frac{11}{5})=(\frac{2}{\sqrt5},-\frac{1}{\sqrt5})\\
(d)&|\widetilde e_1|^2=1\\
(e)&|\widetilde e_2|^2=1\\
(f)&\langle\widetilde e_1,\widetilde e_2\rangle=(\frac{1}{\sqrt5},\frac{2}{\sqrt5})\cdot (\frac{2}{\sqrt5},-\frac{1}{\sqrt5})=0
\end{align}
$$

**Exercise 15**
$$
\begin{align}
(a)&\begin{vmatrix}9 &7\\18 &14\end{vmatrix}=9*14-7*18=0\\
(b)&\langle e_1, w\rangle=9*2+7*4=46\\
(c)&\langle e_2, w\rangle=18*2+14*4=92\\
(d)&ae_1+be_2=(2070,1610)
\end{align}
$$

## 1.5 Systems of Linear Equations
**Exercise 16**
$$
\begin{align}
&\begin{vmatrix}
3 &5 &9\\
-5 &3 &6
\end{vmatrix}\\
=&\begin{vmatrix}
15 &25 &45\\
-15 &9 &18
\end{vmatrix}\\
=&\begin{vmatrix}
3 &5 &9\\
0 &34 &53
\end{vmatrix}
\end{align}
$$

**Exercise 17**
$$x+z=\frac{1}{4}, y=-\frac{1}{4}$$

## 1.6 Bilinear and Quadratic Forms
**Exercise 18**
$$
\begin{align}
(a)&\langle ax+by,z\rangle=(109,41)\cdot(8,4)=1036\\
(b)&a\langle x,z\rangle+b\langle y,z\rangle=5*92+8*72=1036\\
(c)&|ax|^2=45^2+25^2\\
(d)&a^2|x|^2=25*(81+25)
\end{align}
$$

**Exercise 19**
$$
\begin{align}
B(x,y)&=\frac{1}{2}(Q(x+y)-Q(x)-Q(y))\\
&=\frac{1}{2}(4(x_1+y_1)^2+3(x_1+y_1)(x_2+y_2)+6(x_2+y_2)^2-(4x_1^2+3x_1x_2+6x_2^2)-(4y_1^2+3y_1y_2+6y_2^2)\\
&=8x_1y_1+3x_1y_2+3x_2y_1+12x_2y_2
\end{align}
$$

**Exercise 20**
pass

## 1.7 Matrices and Vectors
**Exercise 21**
$$
\begin{align}
(a)&A=\begin{bmatrix}7 &3\\5 &3\end{bmatrix}\\
(b)&Ax=\begin{bmatrix}7 &3\\5 &3\end{bmatrix}\begin{bmatrix}8\\5\end{bmatrix}=\begin{bmatrix}71\\55\end{bmatrix}
\end{align}
$$

**Exercise 22**
$$
\begin{align}
(a)&A=\begin{bmatrix}2 &0\\0 &7\\1 &1\end{bmatrix}\\
(b)&B=\begin{bmatrix}0 &4 &0\\0 &0 &4\\4 &0 &0\end{bmatrix}\\
(c)&BA=\begin{bmatrix}0 &4 &0\\0 &0 &4\\4 &0 &0\end{bmatrix}\begin{bmatrix}2 &0\\0 &7\\1 &1\end{bmatrix}=\begin{bmatrix}0 &28\\4 &4\\8 &0\end{bmatrix}
\end{align}
$$

**Exercise 23**
$$
\begin{align}
(a)&\begin{bmatrix}1 &0\\0 &1\end{bmatrix}\\
(b)&\begin{bmatrix}8 &7\\-14 &24\end{bmatrix}\\
(c)&\begin{bmatrix}8 &7\\-14 &24\end{bmatrix}^{-1}\\
\end{align}
$$

**Exercise 24**
$$
\begin{align}
(a)&A^T=\frac{1}{\sqrt{45}}\begin{bmatrix}6 &-3\\3 &6\end{bmatrix}\\
(b)&A^TA=\frac{1}{45}\begin{bmatrix}6 &-3\\3 &6\end{bmatrix}\begin{bmatrix}6 &3\\-3 &6\end{bmatrix}=\frac{1}{45}\begin{bmatrix}45 &0\\0 &45\end{bmatrix}
\end{align}
$$

**Exercise 25**
pass