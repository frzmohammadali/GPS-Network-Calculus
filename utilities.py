import datetime
import os
import subprocess
import sys
from scipy.optimize import fsolve


def findIntersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)

def powerset_list(lst):
    sublists = [[]]
    for i in range(len(lst) + 1):
        for j in range(i + 1, len(lst) + 1):
            sub = lst[i:j]
            sublists.append(sub)
    return sublists


def powerset_generator(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset_generator(seq[1:]):
            yield [seq[0]] + item
            yield item


def powerset_non_empty_generator(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset_non_empty_generator(seq[1:]):
            yield [seq[0]] + item
            yield item


def distinct_by(seq, idfun=None):
    """
    ## usage sample:
    >>> subsets = distinct_by(powerset(list(range(len(alphas)))), idfun=len)

    :param seq: inout list
    :param idfun: distinct function like len
    :return: generator
    """
    seen = set()
    if idfun is None:
        for x in seq:
            if x not in seen:
                seen.add(x)
                yield x
    else:
        for x in seq:
            x_id = idfun(x)
            if x_id not in seen:
                seen.add(x_id)
                yield x


class ReturnType:
    ITEM = 0
    INDEX = 1

class WeightsMode:
    EQUAL = 'EQUAL'
    RPPS = 'RPPS'
    RANDOM = 'RANDOM'
    FIX = 'FIX'


def length_distinct_subsets(seq, return_type=ReturnType.INDEX, subseteq=False):
    last = []
    yield last
    for ix, item in enumerate(seq):
        if return_type == ReturnType.INDEX:
            last = last + [ix]
            if not subseteq and len(last) == len(seq):
                continue
        elif return_type == ReturnType.ITEM:
            last = last + [item]
            if not subseteq and len(last) == len(seq):
                continue
        else:
            raise Exception("return_type not recognized.")
        yield last


def clear_output():
    # check and make call for specific operating system
    _ = os.system('clear' if os.name == 'posix' else 'cls')


def clear_last_line():
    subprocess.call('', shell=True)
    print("\033[F\033[F\033[K", end="\r")

def filter_generator(fn, gen):
    for item in gen:
        if fn(item):
            yield item

time_of_run = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
def print_write(*s):
    # outputfile = (f"./main.py"
    #              f".run_{time_of_run}.log")
    # with open(os.path.abspath(outputfile), "a") as f:
    #     f.write(" ".join([str(_s) for _s in s]) + '\n')
    print(*s)

def write(*s):
    # outputfile = (f"./main.py"
    #              f".run_{time_of_run}.log")
    # with open(os.path.abspath(outputfile), "a") as f:
    #     f.write(" ".join([str(_s) for _s in s]) + '\n')
    pass
