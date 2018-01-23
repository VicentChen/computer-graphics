# Chapter 4

1. $$\begin{align}
(a*(b*c))[i]&=\Sigma_ja[j](\Sigma_kb[i-j-k]c[k])\\
&=\Sigma_j\Sigma_ka[j]b[i-j-k]c[k]\\
&=\Sigma_k(\Sigma_ja[j]b[i-j-k])c[k]\\
&=((a*b)*c)[i]\\
(a*(b+c))[i]&=\Sigma_ja[i-j](b[i]+c[i])\\
&=\Sigma_ja[i-j]b[i]+\Sigma_ja[i-j]c[i]\\
&=(a*b+a*c)[i]\\
(a*(b*c)))(x)&=\int_ja(j)[\int_kb(i-j-k)c(k)dk]dj\\
&=\int_j\int_ka(j)b(i-j-k)c(k)dkdj\\
&=\int_k(\int_ja[j]b[i-k-j]dj)c(k)dk\\
&=((a*b)*c)(x)\\
(a*(b+c))(i)&=\int_ja(i-j)(b(i)+c(i))dj\\
&=\int_ja(i-j)b(i)dj+\int_ja(i-j)c(i)dj\\
&=(a*b+a*c)(i)\\
\end{align}$$

2. $$\begin{align}
(a*(b*f))(x)&=\Sigma_ia[i](\Sigma_jb[x-i]f[x-i-j])\\
&=\Sigma_i\Sigma_ja[i]b[x-i]f[x-i-j]\\
&=\Sigma_j(\Sigma_ia[i]b[x-i])f[x-i-j]\\
&=((a*b)*f)(x)\\
((a*f)*g)(x)&=\int_i\Sigma_ja[j]f(i-j)g(x-i+j)di\\
&=\Sigma_ja[j]\int_if(i-j)g(x-i+j)di\\
&=(a*(f*g))(x)
\end{align}$$