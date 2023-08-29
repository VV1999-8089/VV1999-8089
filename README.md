# Э-14
# Зинина Валерия
# Урок 8 "Одномерная оптимизация"
# Вариант 10
# Задание 1

import numpy as np
import matplotlib.pyplot as plt

#Вводные данные 

def f(x):
    B = -2.079
    C = 9.454
    D = 7.939
    y = x ** 3 - B * (x ** 2) + C * x - D
    return y
a, b = -1, 5

# Метод деления отрезка пополам

def bisect(f, a, b, eps=0.01):
    x = (a + b) / 2
    while abs(a - b) > eps:
        l = (b - a) / 4
        x1 = a + l
        x2 = b - l
        if f(x2) < f(x):
            a, x = x, x2
        elif f(x1) < f(x):
            b, x = x, x1
        else:
            a, b = x1, x2
    return x, f(x)

# Определение целевой функции (полином)
x, y = bisect(f, a, b)
print ("Методом деления отезка пополам получаем:")
print("x:",x)
print("f(x):",y)
print (" ")


 #Метод деления на три равных отрезка.
 
def trisect(f, a, b, eps=0.01):
    l = b - a
    while l > eps:
        x1 = a + l / 3
        x2 = b - l / 3
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        l = abs(a - b)
    x = (a + b) / 2
    return x, f(x)
 
x, y = trisect(f, a, b)
print ("Методом деления на три равных отрезка получаем:")
print("x:",x)
print("f(x):",y)
print (" ")


#Метод Золотого сечения.

def bisect(f, a, b, eps=0.01):
    x = (a + b) / 2
    while abs(a - b) > eps:
        l = (b - a) / 4
        x1 = a + l
        x2 = b - l
        if f(x2) < f(x):
            a, x = x, x2
        elif f(x1) < f(x):
            b, x = x, x1
        else:
            a, b = x1, x2
    return x, f(x)

x, y = bisect(f, a, b)
print ("Методом золотого сечения получаем:")
print("x:",x)
print("f(x):",y)
print (" ")

 
#Метод чисел Фибоначчи

def Fab_list(Fmax):
    a, b = 0, 1
    Fablist = [a, b] 
    while Fablist[-1] < Fmax:
        a, b = b, a + b
        Fablist.append(b)
    Fablist.pop()  
    return Fablist
 
def fib(f, a, b,n, eps=0.01):
    q = Fab_list(10000)
    x1 =a + (b-a)*q[n-2]/q[n]
    x2 =a + (b-a)*q[n-1]/q[n]
    y1 = f(x1)
    y2 = f(x2)
    while n>=1:
        n-=1
        if y1>y2:
            a = x1
            x1=x2
            x2=b-(x1-a)
            y1=y2
            y2=f(x2)
        else:
            b=x2
            x2=x1
            x1 = a+(b-x2)
            y2=y1
            y1=f(x1)
    x = (x1 + x2) / 2
    return x, f(x)
 
n = 20
x, y = fib(f, a, b,n)
print ("Методом чисел Фибоначчи получаем:")
print("x:",x)
print("f(x):",y)
print (" ")


#Решатели SciPy
from scipy.optimize import minimize

res = minimize(f, 0)
print ("Решателем SciPy получаем:")
print("x:", res.x, "f(x):", res.fun, sep="\n")

# Построение графика
xg = np.arange( -5, 5, 0.1 )
yg = f( xg )
fig, ax1 = plt.subplots() 
ax1.set_xlabel('Ось X') 
ax1.set_ylabel('Ось Y', color = 'black') 
plot_1 = ax1.plot(xg, yg, ".-", color = 'black', label="F(x)" ) 
plt.show()
