"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary Callables.

# Mathematical Callables:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.
# OK!

def mul(x: float, y: float) : 
    return x * y

def id(x: float) : 
    return x

def add(x: float, y: float) :
    return x + y

def neg(x: float) :
    return -x

def lt(x: float, y: float) :
    return x < y

def eq(x: float, y: float) :
    return x == y

def max(x: float, y: float) :
    if x > y:
        return x
    return y

def is_close(x: float, y: float) :
    if abs(x - y) < 0.1:
        return True
    return False

def sigmoid(x: float) :
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    
def relu(x: float) :
    if x < 0:
        return 0
    else:
        return x
    
def log(x: float) :
    return math.log(x)

def exp(x: float) :
    return math.exp(x)

def inv(x: float) :
    if x == 0:
        raise ValueError("Cannot invert zero")
    return 1.0 / x

def inv_back(a: float, b: float) : 
    return - b / (a * a)

def log_back(a: float, b: float) :
    return b / a

def relu_back(a: float, b: float) :
    if a > 0:
        return b
    return 0





# ## Task 0.3

# Small practice library of elementary higher-order Callables.

# Implement the following core Callables
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(func: Callable, iterable: Iterable):
    return (func(item) for item in iterable)

def zipWith(func: Callable, iterable1: Iterable, iterable2: Iterable):
    for x, y in zip(iterable1, iterable2):
        yield func(x, y)

def reduce(func: Callable, iterator: Iterable):
    value = next(iterator)
    for element in iterator:
        value = func(value, element)
    return value

def negList(x: Iterable) :
    return list(map(neg, x))

def addLists(x: list, y: Iterable) :
    return list(zipWith(add, x, y))

def sum(x: Iterable) :
    return reduce(add, x)

def prod(x: Iterable) :
    return reduce(mul, x)