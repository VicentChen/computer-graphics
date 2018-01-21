# CMU-15462-Assigment_0.5
 - Origin Andrew ID: ab1
 - Date: Spetember 5, 2017
 - **Correctness not guaranteed**

# 1 Vector Calculus
## 1.1 Dot and Cross Product
**Exercise 1**
$$
\begin{align}
(a)&u=(4.61,1.95)\\
(b)&v=(4.90,0.99)\\
(c)&l_1l_2cos(|\theta_2-\theta_1|)=5*5*cos(|0.4-0.2|)=24.50\\
(d)&u\cdot v=24.57
\end{align}
$$

**Exercise 2**
$$
\begin{align}
(a)&u=(8,3)\rightarrow(\sqrt{71},21)\\
(b)&v=(8,7)\rightarrow(\sqrt{113},41)\\
(c)&\widetilde u_1=2, \widetilde u_2=\frac{3}{8}\\
(d)&\widetilde v_1=2, \widetilde v_2=\frac{7}{8}\\
(e)&\widetilde u\cdot\widetilde v=(2,\frac{3}{8})\cdot(2,\frac{7}{8})=4\frac{21}{64}\\
(f)&l_1l_2cos(|\theta_2-\theta_1|)=84.16
\end{align}
$$

**Exercise 3**
$$
\begin{align}
(a)&b=4\\
(b)&h=7\\
(c)&A=28\\
(d)&u=(4,0), v=(7,7),|u\times v|=4*7-0=28
\end{align}
$$

**Exercise 4**
$$
\begin{align}
(a)&u\times u=\begin{bmatrix}8*2-2*8\\2*7-7*2\\7*8-8*7\end{bmatrix}=(0,0,0)\\
(b)&u\times v=\begin{bmatrix}8*3-6*2\\2*8-3*7\\7*6-8*8\end{bmatrix}=(12,-5,-22)\\
(c)&v\times u=\begin{bmatrix}6*2-8*3\\3*7-2*8\\8*8-6*7\end{bmatrix}=(-12,5,22)\\
(d)&w\times (u+v)=\begin{bmatrix}3*5-14*7\\7*15-4*5\\4*14-15*3\end{bmatrix}=(-83,85,11)\\
(e)&w\times u+w\times v=\begin{bmatrix}3*2-8*7\\7*7-2*4\\4*8-7*3\end{bmatrix}+\begin{bmatrix}3*3-6*7\\7*8-3*4\\4*6-8*3\end{bmatrix}=(-50,41,11)-(-33,44,0)=(-83,85,11)
\end{align}
$$

**Exercise 5**
$$
\begin{align}
&|u|=1\\
&|v|=5\sqrt2, u\times v=5,u\cdot v=5\\
&|w|=5\sqrt2, u\times 2=5,u\cdot v=5\\
(a)&\theta_{u,v}=arccos(\frac{u\cdot v}{|u||v|})=arccos(\frac{5}{5\sqrt2})=\frac{1}{4}\pi\\
(b)&\theta_{u,w}=arccos(\frac{u\cdot w}{|u||w|})=arccos(\frac{5}{5\sqrt2})=\frac{1}{4}\pi\\
(c)&atan2(\frac{|u\times v|}{u\cdot v})=\frac{1}{4}\pi\\
(d)&atan2(\frac{|u\times w|}{u\cdot w})=\frac{1}{4}\pi\\
(e)&\frac{7}{4}\pi
\end{align}
$$

**Exercise 6**
$$
\begin{align}
(a)&\vec{bc}=(-5,-2,3),\vec{dc}=(0,0,-10)\\
&\vec{bc}\times\vec{dc}=\begin{bmatrix}-2*(-10)-0*3\\3*0-(-10)*(-5)\\-5*0-(-2)*0\end{bmatrix}=(20,-50,0)\\
&|n_1|=10\sqrt{29}\\
(b)&\vec{ac}=(-5,2,3),\vec{ad}=(-5,2,-7)\\
&\vec{ac}\times\vec{ad}=\begin{bmatrix}2*(-7)-2*3\\3*(-5)-(-7)*(-5)\\-5*2-(-5)*2\end{bmatrix}=(-20,-50,0)\\
&|n_2|=10\sqrt{29}\\
(c)&w=(0,0,10)\\
&\theta=atan2(\frac{|n_1\times n_2|}{n_1\cdot n_2})=atan2(\frac{2000}{2100})=0.76
\end{align}
$$

**Exericise 7**
$$
\begin{align}
(a)&u\cdot v=18+36+0=54\\
(b)&|u\times v|=|\begin{bmatrix}6*0-6*2\\2*6-0*3\\3*6-6*6\end{bmatrix}|=|(-12,12,-18)|=6\sqrt{17}\\
(c)&\theta=arccos(\frac{u\cdot v}{|u||v|})=arccos(\frac{54}{7*6\sqrt2})=0.43\\
(d)&cot\theta=2.18\\
(e)&\frac{u\cdot v}{u\times v}=\frac{54}{6\sqrt{17}}=2.18
\end{align}
$$

**Exercise 8**
$$
\begin{align}
(a)&u\times a=\begin{bmatrix}2*2-7*6\\6*4-2*5\\5*7-4*2\end{bmatrix}=(-38,14,27)\\
(b)&\hat u=\begin{bmatrix}0 &-6 &2\\6 &0 &-5\\-2 &5 &0\end{bmatrix}\\
(c)&\hat u^Ta=\begin{bmatrix}0 &6 &-2\\-6 &0 &5\\2 &-5 &0\end{bmatrix}\begin{bmatrix}4\\7\\2\end{bmatrix}=(38,-14,-27)
\end{align}
$$

**Exericse 9**
$$
\begin{align}
(a)&u\cdot (v\times u)=0\\
(b)&u\cdot (v\times w)=90\\
(c)&v\cdot (u\times w)=126\\
(d)&u\times v-(v\times w)=(-66,84,-48)\\
(e)&v(u\cdot w)-w(u\dot v)=(-66, 84, -48)\\
(f)&v\times(w\times u)+w\times(u\times v)=(-66, 84, -48)
\end{align}
$$

## 1.2 Vector Fields
**Exercies 10**
$$
\begin{align}
(a)&Z(u)=(6cos(9*0.9)+5,5*0.5^2+7(0.5^2+0.9^2))=(3.54,20.85)\\
(b)&8Z(u)=8*(6cos(9*0.9)+5,5*0.8^2+7*(0.8^2+0.9^2))=(3.54,18.15)
\end{align}
$$

**Exericise 11**
$$
\begin{align}
(a)&(X\cdot X)(u)=4*9^2+16*6^2+1*1=901\\
(b)&(X\cdot \gamma)(u)=-8u_1u_2+8u_2u_1+u_3=9\\
(c)&(X\times \gamma)(u)=\begin{bmatrix}4*7*6-2*5\\-4*7-2*5*6\\4*5*5+16*7*7\end{bmatrix}=(158,-88,884)
\end{align}
$$

## 1.3 Gradient, Divergence and Curl
**Exercise 12**
$$
\begin{align}
(a)&f(9,7)=2*9^2+7*cos(4*7)=155.2\\
(b)&g(4,3,5)=9*4*3+7*3*5+2*5*4=253\\
(c)&h(6)=log(6)+9/6=2.28
\end{align}
$$

**Exercise 13**
$$
\begin{align}
(a)&f(x)=x_1y_1+x_2y_2\\
&(\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2})=(y1, y2)=(5, 8)\\
(b)&D_uf(x_0)\\
&=lim_{\varepsilon\rightarrow0}\frac{f((2,8)+\varepsilon(4,5))-f((2,8))}{\varepsilon}\\
&=60\\
(c)&(4,5)\\
(d)&pass\\
(e)&pass
\end{align}
$$

**Exercise 14**
$$
\begin{align}
(a)&f(x_0)=cos(7),f'(x_0)=-sin(7),f''(x_0)=-cos(7)\\
(b)&\hat f(7.070)=0.706\\
(c)&\hat f(7.070)-f(7.070)=-3.83*10^{-5}\\
(d)&\nabla g=(-sin(x)sin(y),cos(x)cos(y))\\
&\nabla g(3,5)=(0.13,-0.28)\\
(e)&\nabla^2g=\begin{bmatrix}-cos(x)sin(y) &-sin(x)cos(y)\\-sin(x)cos(y) &-cos(x)sin(y)\end{bmatrix}\\
&\nabla^2g(3,5)=\begin{bmatrix}-0.95 &-0.04\\-0.04 &-0.95\end{bmatrix}\\
(f)&\hat g(x)=0.95+(x-(3,5))^T(0.13,-0.28)+\frac{1}{2}(x-(3,5))^T\begin{bmatrix}-0.95 &-0.04\\-0.04
&-0.95\end{bmatrix}(x-(3,5))\\
pass\\
(g)pass
\end{align}
$$

**Exercise 15**
$$
\begin{align}
(a)&\nabla\cdot X=(0, 8u)=(0,48)\\
(b)&\nabla\cdot \gamma=(-8v,8v)=(-24,24)\\
(c)&\nabla\times X=8v-8v=0\\
(d)&\nabla\times \gamma=-8u-0=-48\\
(e)&\nabla\cdot Z+\nabla\times Z=0
\end{align}
$$

**Exercise 16**
$$
\begin{align}
(a)&\nabla f=(5v,5u)=(30,30)\\
(b)&divZ=\nabla\cdot Z=12u+10v=132\\
(c)&(\nabla f)\cdot Z=(5v,5u)\cdot(6u^2,5v^2)=30u^2v+25uv^3=11880\\
(d)&f(\nabla\cdot Z)=132*5*u*v=23760\\
(e)&(\nabla f)\cdot Z+f(\nabla\cdot Z)=35640\\
(f)&fZ=(30u^3v,25uv^3)=(38880,32400)\\
(g)&\nabla\cdot(fZ)=90u^2v+75uv^2=35640
\end{align}
$$

**Exercise 17**
$$
\begin{align}
(a)&g(\nabla\times U)+(\nabla g)\times U\\
&=(0,0,0)+(620,546,665)\times (0,1,8)\\
&=(3703,-4960,620)\\
(b)&(3703,-4960,620)
\end{align}
$$

**Exercise 18**
pass
