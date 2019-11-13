---
layout: post
title: numba详解
categories: Python
tags: Python
excerpt: Python 比较慢，特别是在写循环的时候，运行速度慢的非常明显，这里介绍一个Python中加速的包。
mathjax: false
date: 2018-10-10
---


* content
{:toc}


我们都知道 Python 比较慢，但很多时候我们都不知道为什么，但也只能模模糊糊地感受到这么两点：

* Python 太动态了

* 如果能事先编译一下 Python，让它静态一点，速度应该就会上来

于是我们就有了 cython。然而 cython 毕竟不是原生的 Python 代码，使用起来还是有诸多不便的。为此，numba 就成了一个功能强大又容易上手的替代选择。下面我们就先来看一下它的基本用法，在最后我们则会用卷积神经网络（CNN）的卷积和池化操作来直观感受一下其威力

（注意：以下代码的运行环境均为 Jupyter Notebook、Python3.6.1）

## 使用 jit 加速 Python 低效的 for 语句

jit 的全称是 Just-in-time，在 numba 里面则特指 Just-in-time compilation（即时编译）。它背后的原理我们就不细说了，总之我们看到“编译”两个字大概就能感受到它是干什么的对吧（喂

那么来看一个简单的栗子——给数组中的每个数加上一个常数 c：

```python
import numba as nb
import numpy as np

# 普通的 for
def add1(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs

# list comprehension
def add2(x, c):
    return [xx + c for xx in x]

# 使用 jit 加速后的 for
@nb.jit(nopython=True)
def add_with_jit(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs

y = np.random.random(10**5).astype(np.float32)
x = y.tolist()

assert np.allclose(add1(x, 1), add2(x, 1), add_with_jit(x, 1))
%timeit add1(x, 1)
%timeit add2(x, 1)
%timeit add_with_jit(x, 1)
print(np.allclose(wrong_add(x, 1), 1))

```

以下是程序运行结果：

```numpy
9.92 ms ± 188 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
5.77 ms ± 347 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
3.48 ms ± 171 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

需要注意的是：

* numba不支持 list comprehension，详情可参见[这里](https://link.zhihu.com/?target=https%3A//github.com/numba/numba/issues/504)

* jit能够加速的不限于for，但一般而言加速for会比较常见、效果也比较显著。我在我实现的numpy版本的卷积神经网络（CNN）中用了jit后、可以把代码加速 20 倍左右。具体代码可以参见[这里](https://link.zhihu.com/?target=https%3A//github.com/carefree0910/MachineLearning/blob/master/NN/Basic/Layers.py%23L9)，不过如果不想看源代码的话，可以参见[CNN.ipynb](https://link.zhihu.com/?target=https%3A//github.com/carefree0910/MachineLearning/blob/master/Notebooks/numba/zh-cn/CNN.ipynb)，我在其中做了一些相应的、比较简单的实验

* jit会在某种程度上“预编译”你的代码，这意味着它会在某种程度上固定住各个变量的数据类型；所以在jit下定义数组时，如果想要使用的是float数组的话，就不能用[0] * len(x)定义、而应该像上面那样在0后面加一个小数点：[0.] * len(x)

## 使用 vectorize 实现 numpy 的 Ufunc 功能

虽然jit确实能让我们代码加速不少，但比之numpy的Ufunc还是要差很多：

```python
assert np.allclose(y + 1, add_with_jit(x, 1))
%timeit add_with_jit(x, 1)
%timeit y + 1
```

结果将会是：

```text
3.76 ms ± 292 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
19.8 µs ± 426 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

可以看到几乎有 200 倍的差距，这当然是无法忍受的。为此，我们可以用vectorize来定义出类似于Ufunc的函数：

```python
@nb.vectorize(nopython=True)
def add_with_vec(yy, c):
    return yy + c

assert np.allclose(y + 1, add_with_vec(y, 1), add_with_vec(y, 1.))
%timeit add_with_vec(y, 1)
%timeit add_with_vec(y, 1.)
%timeit y + 1
%timeit y + 1.

```

上述程序的运行结果将会是：

```text
72.5 µs ± 3.46 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
64.2 µs ± 1.9 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
24.6 µs ± 1.81 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
25.3 µs ± 1.61 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

虽然还是慢了 2 倍左右，但已经好很多了

然后有几点需要注意的地方：

* vectorize下的函数所接受的参数都是一个个的数而非整个数组。所以上述add_with_vec的参数yy其实是输入数组y中的元素，而不是y本身。更详细的说明可以参见[官方文档](https://link.zhihu.com/?target=http%3A//numba.pydata.org/numba-doc/0.23.0/user/vectorize.html)）

* 可以看到当常数 c 是整数和是浮点数时、速度是不同的。个人猜测这是因为若常数 c 为整数，那么实际运算时需要将它转化为浮点数，从而导致速度变慢

* 上述代码中我们没有显式地定义函数的参数类型和返回类型，但我们可以预先定义。比如说，如果我确定常数 c 就是整数的话，我就可以这样写：

```python
@nb.vectorize("float32(float32, int32)", nopython=True)
def add_with_vec(yy, c):
    return yy + c

```

而如果我确定常数 c 就是浮点数的话，我就可以这样写：

```python
@nb.vectorize("float32(float32, float32)", nopython=True)
def add_with_vec(yy, c):
    return yy + c

```

而如果我确定常数 c 不是整数就是浮点数的话（这个人好啰嗦！），我就可以这样写：

```python
@nb.vectorize([
    "float32(float32, int32)",
    "float32(float32, float32)"
], nopython=True)
def add_with_vec(yy, c):
    return yy + c

```

注意，float32 和 float64、int32 和 int64 是不同的，需要小心

此外，vectorize最炫酷的地方在于，它可以“并行”：

```python
@nb.vectorize("float32(float32, float32)", target="parallel", nopython=True)
def add_with_vec(y, c):
    return y + c

assert np.allclose(y+1, add_with_vec(y,1.))
%timeit add_with_vec(y, 1.)
%timeit y + 1
```

虽说在普通的 Python3.6.1 下、运行结果将如下：

```text
73.5 µs ± 4.22 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
21.2 µs ± 734 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

似乎还变慢了；不过如果使用 Intel Distribution for Python 的话，会发现parallel版本甚至会比numpy原生的版本要稍快一些

那么是否有用parallel总会更好的栗子呢？当然是有的：

```python
# 将数组所有元素限制在某个区间[a, b]内
# 小于 a 则置为 a，大于 b 则置为 b
# 经典应用：ReLU

@nb.vectorize("float32(float32, float32, float32)", target="parallel", nopython=True)
def clip_with_parallel(y, a, b):
    if y < a:
        return a
    if y > b:
        return b
    return y

@nb.vectorize("float32(float32, float32, float32)", nopython=True)
def clip(y, a, b):
    if y < a:
        return a
    if y > b:
        return b
    return y

assert np.allclose(np.clip(y, 0.1, 0.9), clip(y, 0.1, 0.9), clip_with_parallel(y, 0.1, 0.9))
%timeit clip_with_parallel(y, 0.1, 0.9)
%timeit clip(y, 0.1, 0.9)
%timeit np.clip(y, 0.1, 0.9)

```

上述程序的运行结果将会是：

```text
95.2 µs ± 5.6 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
104 µs ± 4.52 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
377 µs ± 62 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

```

这个栗子中的性能提升就是实打实的了。总之，使用parallel时不能一概而论，还是要做些实验

需要指出的是，vectorize中的参数target一共有三种取值：cpu（默认）、parallel和cuda。关于选择哪个取值，官方文档上有很好的说明：

> A general guideline is to choose different targets for different data sizes and algorithms. The “cpu” target works well for small data sizes (approx. less than 1KB) and low compute intensity algorithms. It has the least amount of overhead. The “parallel” target works well for medium data sizes (approx. less than 1MB). Threading adds a small delay. The “cuda” target works well for big data sizes (approx. greater than 1MB) and high compute intensity algorithms. Transfering memory to and from the GPU adds significant overhead.

## 使用 jit(nogil=True) 实现高效并发（多线程）

我们知道，Python 中由于有 [GIL](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/GIL) 的存在，所以多线程用起来非常不舒服。不过 numba 的 jit 里面有一项参数叫 nogil，想来聪明的观众老爷们都猜到了它是干什么的了……

下面就来看一个栗子：

```python
import math
from concurrent.futures import ThreadPoolExecutor

# 计算类似于 Sigmoid 的函数
def np_func(a, b):
    return 1 / (a + np.exp(-b))

# 参数中的 result 代表的即是我们想要的结果，后同
# 第一个 kernel，nogil 参数设为了 False
@nb.jit(nopython=True, nogil=False)
def kernel1(result, a, b):
    for i in range(len(result)):
        result[i] = 1 / (a[i] + math.exp(-b[i]))

# 第二个 kernel，nogil 参数设为了 True
@nb.jit(nopython=True, nogil=True)
def kernel2(result, a, b):
    for i in range(len(result)):
        result[i] = 1 / (a[i] + math.exp(-b[i]))

def make_single_task(kernel):
    def func(length, *args):
        result = np.empty(length, dtype=np.float32)
        kernel(result, *args)
        return result
    return func

def make_multi_task(kernel, n_thread):
    def func(length, *args):
        result = np.empty(length, dtype=np.float32)
        args = (result,) + args
        # 将每个线程接受的参数定义好
        chunk_size = (length + n_thread - 1) // n_thread
        chunks = [[arg[i*chunk_size:(i+1)*chunk_size] for i in range(n_thread)] for arg in args]
        # 利用 ThreadPoolExecutor 进行并发
        with ThreadPoolExecutor(max_workers=n_thread) as e:
            for _ in e.map(kernel, *chunks):
                pass
        return result
    return func

length = 10 ** 6
a = np.random.rand(length).astype(np.float32)
b = np.random.rand(length).astype(np.float32)

nb_func1 = make_single_task(kernel1)
nb_func2 = make_multi_task(kernel1, 4)
nb_func3 = make_single_task(kernel2)
nb_func4 = make_multi_task(kernel2, 4)

rs_np = np_func(a, b)
rs_nb1 = nb_func1(length, a, b)
rs_nb2 = nb_func2(length, a, b)
rs_nb3 = nb_func3(length, a, b)
rs_nb4 = nb_func4(length, a, b)
assert np.allclose(rs_np, rs_nb1, rs_nb2, rs_nb3, rs_nb4)
%timeit np_func(a, b)
%timeit nb_func1(length, a, b)
%timeit nb_func2(length, a, b)
%timeit nb_func3(length, a, b)
%timeit nb_func4(length, a, b)
```

这个栗子有点长，不过我们只需要知道如下两点即可：

* make_single_task和make_multi_task分别生成单线程函数和多线程函数

* 生成的函数会调用相应的kernel来完成计算

上述程序的运行结果将会是：

```text
14.9 ms ± 538 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
8.32 ms ± 259 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
10.2 ms ± 368 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
8.25 ms ± 279 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
4.68 ms ± 114 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

```

一般来说，数据量越大、并发的效果越明显。反之，数据量小的时候，并发很有可能会降低性能

## numba 的应用实例 —— 卷积与池化

如果只想看效果的话倒没什么关系，不过如果想知道我具体在干什么的话，可以参见[这篇文章](https://zhuanlan.zhihu.com/p/26657869)

首先是卷积操作：

```python
import numba as nb
import numpy as np

# 普通的卷积
def conv_kernel(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(n_filters):
                    rs[i, q, j, p] += np.sum(w[q] * window)
    return rs

# 简单地加了个 jit 后的卷积
@nb.jit(nopython=True)
def jit_conv_kernel(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(n_filters):
                    rs[i, q, j, p] += np.sum(w[q] * window)

# 卷积操作的封装
def conv(x, w, kernel, args):
    n, n_filters = args[0], args[4]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    kernel(x, w, rs, *args)
    return rs

# 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
x = np.random.randn(64, 3, 28, 28).astype(np.float32)
# 16 个 5 x 5 的 kernel
w = np.random.randn(16, x.shape[1], 5, 5).astype(np.float32)

n, n_channels, height, width = x.shape
n_filters, _, filter_height, filter_width = w.shape
out_h = height - filter_height + 1
out_w = width - filter_width + 1
args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)

print(np.linalg.norm((conv(x, w, conv_kernel, args) - conv(x, w, jit_conv_kernel, args)).ravel()))
%timeit conv(x, w, conv_kernel, args)
%timeit conv(x, w, jit_conv_kernel, args)

```

上述程序的运行结果将会是：

```text
0.00112681
3.63 s ± 194 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
300 ms ± 20.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

可以看到，仅仅是加了一个jit、速度就直接提升了十多倍

有细心的观众老爷可能已经发现，我这里没有使用np.allclose；这是因为卷积涉及到的运算太多，仅仅是将数组的dtype从float64变成float32、精度就会大幅下降，所以使用np.allclose的话会过不了assert

同时需要特别注意的是，使用jit和使用纯numpy进行编程的很大一点不同就是，不要畏惧用for；事实上一般来说，代码“长得越像 C”、速度就会越快：

```python
@nb.jit(nopython=True)
def jit_conv_kernel2(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(n_filters):
                    for r in range(n_channels):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]

assert np.allclose(conv(x, w, jit_conv_kernel, args), conv(x, w, jit_conv_kernel, args))
%timeit conv(x, w, jit_conv_kernel, args)
%timeit conv(x, w, jit_conv_kernel2, args)
```

那么程序的运行结果将会是：

```python
281 ms ± 12.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
66.2 ms ± 2.32 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

可以看到这又快了 3 倍左右

接下来是池化操作（我们选用的是 MaxPool）：

```python
# 普通的 MaxPool
def max_pool_kernel(x, rs, *args):
    n, n_channels, pool_height, pool_width, out_h, out_w = args
    for i in range(n):
        for j in range(n_channels):
            for p in range(out_h):
                for q in range(out_w):
                    window = x[i, j, p:p+pool_height, q:q+pool_width]
                    rs[i, j, p, q] += np.max(window)

# 简单地加了个 jit 后的 MaxPool
@nb.jit(nopython=True)
def jit_max_pool_kernel(x, rs, *args):
    n, n_channels, pool_height, pool_width, out_h, out_w = args
    for i in range(n):
        for j in range(n_channels):
            for p in range(out_h):
                for q in range(out_w):
                    window = x[i, j, p:p+pool_height, q:q+pool_width]
                    rs[i, j, p, q] += np.max(window)

# 不惧用 for 的、“更像 C”的 MaxPool
@nb.jit(nopython=True)
def jit_max_pool_kernel2(x, rs, *args):
    n, n_channels, pool_height, pool_width, out_h, out_w = args
    for i in range(n):
        for j in range(n_channels):
            for p in range(out_h):
                for q in range(out_w):
                    _max = x[i, j, p, q]
                    for r in range(pool_height):
                        for s in range(pool_width):
                            _tmp = x[i, j, p+r, q+s]
                            if _tmp > _max:
                                _max = _tmp
                    rs[i, j, p, q] += _max

# MaxPool 的封装
def max_pool(x, kernel, args):
    n, n_channels = args[:2]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    kernel(x, rs, *args)
    return rs

pool_height, pool_width = 2, 2
n, n_channels, height, width = x.shape
out_h = height - pool_height + 1
out_w = width - pool_width + 1
args = (n, n_channels, pool_height, pool_width, out_h, out_w)

assert np.allclose(max_pool(x, max_pool_kernel, args), max_pool(x, jit_max_pool_kernel, args))
assert np.allclose(max_pool(x, jit_max_pool_kernel, args), max_pool(x, jit_max_pool_kernel2, args))
%timeit max_pool(x, max_pool_kernel, args)
%timeit max_pool(x, jit_max_pool_kernel, args)
%timeit max_pool(x, jit_max_pool_kernel2, args)

```

上述程序的运行结果将会是：

```text
586 ms ± 38 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
8.25 ms ± 526 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
1.4 ms ± 57 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

可以看到最快的比最慢的要快整整 **400** 倍有多

