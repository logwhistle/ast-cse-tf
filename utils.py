# -*- coding: utf-8 -*-
# contact: log_whistle@163.com
from collections.abc import MutableSet
from collections import OrderedDict


SYMINDEX = 0


def debug(obj, msg=''):
    if False:
        print('DEBUG', msg, obj)


def get_sym():
    global SYMINDEX
    sym = 'sub%04d' % SYMINDEX
    SYMINDEX += 1
    return sym


class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        if iterable:
            self.map = OrderedDict((item, None) for item in iterable)
        else:
            self.map = OrderedDict()

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        self.map[key] = None

    def discard(self, key):
        self.map.pop(key)

    def pop(self, last=True):
        return self.map.popitem(last=last)[0]

    def __iter__(self):
        yield from self.map.keys()

    def __repr__(self):
        if not self.map:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.map.keys()))

    def intersection(self, other):
        result = []
        for val in self:
            if val in other:
                result.append(val)
        return self.__class__(result)

    def difference(self, other):
        result = []
        for val in self:
            if val not in other:
                result.append(val)
        return self.__class__(result)

    def update(self, iterable):
        for val in iterable:
            self.add(val)