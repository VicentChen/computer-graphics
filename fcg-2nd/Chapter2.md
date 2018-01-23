# Chapter 2

 1.  - 32位浮点数：符号位1bit, 指数位8bits, 小数位23bits
     - 小数位足够表示$2^{23}$个数字
     - 指数位能够表示$2^8-1=255$个位移
     - 符号位能够表示2种数字
     - $2^{24}*255$
 - 64位浮点数：符号位1bit, 指数位11bits, 小数位52bits
     - 小数位足够表示$2^{52}$个数字
     - 指数位能够表示$2^11-1=2047$个位移
     - 符号位能够表示2种数字
     - $2^{53}*2047$
 2. 不可能，二者间元素个数不同，无法一一对应。
 3. $$Cube=X(0,1)\times Y(0,1)\times Z(0,1)$$
 4. $$log(b,x)=ln(x)/ln(b)$$
 5. $$4x^2-6x+9=(2x-3)^2\rightarrow x=1.5$$
 6. $$\Delta=B^2-4AC\\x=\frac{-B\pm\sqrt\Delta}{2A}$$
 7. $$a=(0,1,0),b=(0,1,0),c=(0,0,1)\\a\times(b\times c)=(0,0,-1)\\(a\times b)\times c=(0,0,0)$$
 8. $$u=\frac{a}{|a|}, v=\frac{(a\times b)\times c}{|(a\times b)\times c|}$$
 9. $$\nabla f=(2x,1,9z^2)$$
 10. $$\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}x_0+acos\theta\\y_0+bsin\theta\end{bmatrix}$$
 11. $$(p-a)\cdot((c-a)\times(b-a))=0\\\begin{bmatrix}x\\y\\z\end{bmatrix}=\begin{bmatrix}x_a+u(b-a)+v(b-a)\\y_a+u(b-a)+v(b-a)\\z_a+u(b-a)+v(b-a)\end{bmatrix}\\(c-a)\times(b-a)$$
 12. $$(y_{a_0}-y_{a_1},x_{a_0}-x_{a_1})\cdot(y_{b_0}-y_{b_1},x_{b_0}-x_{b_1})\not= 0$$
 13. $$\begin{bmatrix}x_p\\y_p\end{bmatrix}=\begin{bmatrix}x_b-x_a&x_c-x_a\\y_b-y_a&y_c-y_a\end{bmatrix}\begin{bmatrix}\beta\\\gamma\end{bmatrix}+\begin{bmatrix}x_a\\y_a\end{bmatrix}$$