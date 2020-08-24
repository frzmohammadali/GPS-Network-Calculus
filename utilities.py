import os
import subprocess
import sys


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
    EQUAL = 0
    RPPS = 1
    RANDOM = 2


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
