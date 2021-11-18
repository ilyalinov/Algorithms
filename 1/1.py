import numpy as np
import time
import matplotlib.pyplot as plt
import decimal


def gen_vec(n):
    return np.random.randint(100, size=n)

def gen_mat(n):
    return np.random.randint(100, size=(n, n))

def const(v):
    return 1

def sum(v):
    return np.sum(v)

def mult(v):
    return np.prod(v)

def p(v):
    i = 0
    decimal.getcontext().prec = 10
    res = decimal.Decimal(0)
    for c in v:
        res += decimal.Decimal(float(c)) * decimal.Decimal(1.5) ** decimal.Decimal(float(i))
        i += 1
    return res

def p_gorner(v):
    res = decimal.Decimal(0)
    for i in range(len(v) - 1, -1, -1):
        res *= decimal.Decimal(1.5)
        res += decimal.Decimal(float(v[i]))
    return res

def bubble_sort(v):
    n = len(v)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if v[j] > v[j + 1]:
                v[j], v[j + 1] = v[j + 1], v[j]
                swapped = True
        if swapped == False:
            break

def timsort(v):
    return np.sort(v, kind='stable')

def quicksort(v):
    return np.sort(v, kind='quicksort')

def matmul(A, B):
    return np.matmul(A, B)

def run(fun):
    t = []
    for i in range(1, 2001):
        res = 0.0
        v = gen_vec(i)
        for j in range(5):
            start = time.time()
            fun(v)
            stop = time.time()
            res += stop - start

        t.append(res / 5)
        print(i)

    return t

def run_mat(fun):
    t = []
    for i in range(1, 301):
        res = 0.0
        A = gen_mat(i)
        B = gen_mat(i)
        for j in range(5):
            start = time.time()
            fun(A, B)
            stop = time.time()
            res += stop - start

        t.append(res / 5)

    return t

# n = 3
# v = gen_vec(n)
# print(v)
# A = gen_mat(n)
# B = gen_mat(n)
# print(A)
# print(B)
# print(const(v))
# print(sum(v))
# print(mult(v))
# print(p(v))
# print(p_gorner(v))
# bubble_sort(v)
# print(v)
# v = timsort(v)
# print(v)
# v = quicksort(v)
# print(v)
# print(matmul(A, B))

def f1(x, y):
    return y[0] + x * np.log2(x) * (y[len(y) - 1] - y[0]) / (np.log2(2000) * 2000)

def f2(x, y):
    return x * (y[999]/ (1000))

def f3(x, y):
    return x * x * (y[len(y) - 1]) / (2000 * 2000)

def f4(x, y):
    return y[0] + x * 0

def f5(x, y):
    return y[0] + x * np.log(x) * (y[len(y) - 1] - y[0]) / (np.log(2000) * 2000)

def f6(x, y):
    return x**3 * (y[149]/ 150**3)

# PLOTS
#
#
x = np.linspace(1, 2001, 2000)

# quicksort
t = run(quicksort)
k = 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f1(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()

# timsort
t = run(timsort)
k += 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f1(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()

# bubble sort
#t = run(bubble_sort)
k += 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f3(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()

# const
t = run(const)
k += 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f4(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()

# sum
t = run(sum)
k += 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f2(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()

# sum
t = run(mult)
k += 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f2(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()

# P(1.5)
#t = run(p)
k += 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f5(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()


# P(1.5) Gorner
#t = run(p_gorner)
k += 1

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f2(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.clf()

# matrices multiplication
k += 1
t = run_mat(matmul)

x = np.linspace(1, 301, 300)

plt.plot(x, t, label='empirical complexity')
plt.plot(x, f6(x, t), label='theoretical complexity')
plt.ylabel('time, seconds')
plt.xlabel('n')
plt.legend()
plt.savefig('plot' + str(k) + '.png')
plt.show()
# print(t)