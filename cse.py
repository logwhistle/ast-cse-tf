# -*- coding: utf-8 -*-
# contact: log_whistle@163.com
from collections import defaultdict
from collections import OrderedDict

import ast_all as ast
from utils import debug, get_sym, OrderedSet


ELIMINATE_ALL = True  # defined features间是否互相依赖
# 若为True，需要lambda解析json的时候使用ordered_map


def tree_cse(exprs, mod='normal'):
    """return: expressions, defined_expressions"""
    opt_subs = opt_cse(exprs)
    debug(opt_subs, 'opt_subs')
    to_eliminate = set()
    seen_subexp = set()

    def _find_repeated(expr):
        debug(expr, 'find_repeated_expr_entry')
        if expr.op.type == ast.VAR:
            return
        if expr in seen_subexp:
            if expr.op.type == ast.NOT and len(expr.args) == 1 and expr.args[0].op.type == ast.VAR and mod == 'math':
                print(expr.op.type, expr.args)
                return
            return to_eliminate.add(expr)
        seen_subexp.add(expr)
        if expr in opt_subs:
            expr = opt_subs[expr]
        list(map(_find_repeated, expr.args))

    for e in exprs:
        _find_repeated(e)
    debug(to_eliminate, 'to_eliminate')

    subs = dict()
    replacements = dict()

    def _rebuild(expr):
        if expr.op == ast.VAR:
            return expr

        if expr in subs:
            return subs[expr]

        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        if not ELIMINATE_ALL:
            # 有子节点要被折叠，且当前节点也要被折叠
            if any(arg in to_eliminate for arg in expr.args) and orig_expr in to_eliminate:
                sym = ast.Tree(ast.Token(ast.VAR, get_sym()), [])
                subs[orig_expr] = sym
                replacements[sym] = orig_expr
                return sym

        new_args = list(map(_rebuild, expr.args))
        if new_args != expr.args:
            new_expr = ast.Tree(expr.op, new_args)
        else:
            new_expr = expr

        if orig_expr in to_eliminate:
            sym = ast.Tree(ast.Token(ast.VAR, get_sym()), [])
            subs[orig_expr] = sym
            replacements[sym] = new_expr
            return sym
        else:
            return new_expr

    reduced_exprs = []
    for e in exprs:
        reduced_e = _rebuild(e)
        reduced_exprs.append(reduced_e)
    debug(replacements, 'replacement')
    debug(reduced_exprs, 'reduced_exprs')
    return replacements, reduced_exprs


def opt_cse(exprs):
    """Find optimization opportunities"""
    opt_subs = dict()

    ors = OrderedSet()
    ands = OrderedSet()
    seen_subexp = set()

    def _find_opts(expr):
        if expr.op.type == ast.VAR:
            return
        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)
        list(map(_find_opts, expr.args))
        if expr.op.type == ast.OR:
            ors.add(expr)
        if expr.op.type == ast.AND:
            ands.add(expr)

    for e in exprs:
        _find_opts(e)

    debug(ors, 'ors')
    debug(ands, 'ands')
    match_common_args(ast.OR, ors, opt_subs)
    match_common_args(ast.AND, ands, opt_subs)
    return opt_subs


def match_common_args(func_class, funcs, opt_subs):
    """依据交换律找到公共参数"""
    funcs = sorted(funcs, key=lambda func: len(func.args))
    arg_tracker = FuncArgTracker(funcs)
    changed = OrderedSet()
    for i in range(len(funcs)):
        common_arg_candidates_counts = arg_tracker.get_common_arg_candidates(
                arg_tracker.func_to_argset[i], min_func_i=i + 1)

        # Sort the candidates in order of match size.
        # This makes us try combining smaller matches first.
        common_arg_candidates = OrderedSet(sorted(
                common_arg_candidates_counts.keys(),
                key=lambda k: (common_arg_candidates_counts[k], k)))

        while common_arg_candidates:
            j = common_arg_candidates.pop(last=False)

            com_args = arg_tracker.func_to_argset[i].intersection(
                    arg_tracker.func_to_argset[j])

            if len(com_args) <= 1:
                # This may happen if a set of common arguments was already
                # combined in a previous iteration.
                continue

            # For all sets, replace the common symbols by the function
            # over them, to allow recursive matches.

            diff_i = arg_tracker.func_to_argset[i].difference(com_args)
            if diff_i:
                # com_func needs to be unevaluated to allow for recursive matches.
                com_func = ast.Tree(ast.Token(func_class), arg_tracker.get_args_in_value_order(com_args))
                com_func_number = arg_tracker.get_or_add_value_number(com_func)
                arg_tracker.update_func_argset(i, diff_i | OrderedSet([com_func_number]))
                changed.add(i)
            else:
                # Treat the whole expression as a CSE.
                #
                # The reason this needs to be done is somewhat subtle. Within
                # tree_cse(), to_eliminate only contains expressions that are
                # seen more than once. The problem is unevaluated expressions
                # do not compare equal to the evaluated equivalent. So
                # tree_cse() won't mark funcs[i] as a CSE if we use an
                # unevaluated version.
                com_func_number = arg_tracker.get_or_add_value_number(funcs[i])

            diff_j = arg_tracker.func_to_argset[j].difference(com_args)
            arg_tracker.update_func_argset(j, diff_j | OrderedSet([com_func_number]))
            changed.add(j)

            for k in arg_tracker.get_subset_candidates(
                    com_args, common_arg_candidates):
                diff_k = arg_tracker.func_to_argset[k].difference(com_args)
                arg_tracker.update_func_argset(k, diff_k | OrderedSet([com_func_number]))
                changed.add(k)

        if i in changed:
            opt_subs[funcs[i]] = ast.Tree(
                ast.Token(func_class), arg_tracker.get_args_in_value_order(arg_tracker.func_to_argset[i]))

        arg_tracker.stop_arg_tracking(i)


class FuncArgTracker:
    """
    A class which manages a mapping from functions to arguments and an inverse
    mapping from arguments to functions.
    """

    def __init__(self, funcs):
        # To minimize the number of symbolic comparisons, all function arguments
        # get assigned a value number.
        self.value_numbers = {}
        self.value_number_to_value = []

        # Both of these maps use integer indices for arguments / functions.
        self.arg_to_funcset = []
        self.func_to_argset = []

        for func_i, func in enumerate(funcs):
            func_argset = OrderedSet()

            for func_arg in func.args:
                arg_number = self.get_or_add_value_number(func_arg)
                func_argset.add(arg_number)
                self.arg_to_funcset[arg_number].add(func_i)
            self.func_to_argset.append(func_argset)

    def get_args_in_value_order(self, argset):
        """
        Return the list of arguments in sorted order according to their value
        numbers.
        """
        return [self.value_number_to_value[argn] for argn in sorted(argset)]

    def get_or_add_value_number(self, value):
        """
        Return the value number for the given argument.
        """
        nvalues = len(self.value_numbers)
        value_number = self.value_numbers.setdefault(value, nvalues)
        if value_number == nvalues:
            self.value_number_to_value.append(value)
            self.arg_to_funcset.append(OrderedSet())
        return value_number

    def stop_arg_tracking(self, func_i):
        """
        Remove the function func_i from the argument to function mapping.
        """
        for arg in self.func_to_argset[func_i]:
            self.arg_to_funcset[arg].remove(func_i)

    def get_common_arg_candidates(self, argset, min_func_i=0):
        """Return a dict whose keys are function numbers. The entries of the dict are
        the number of arguments said function has in common with
        ``argset``. Entries have at least 2 items in common.  All keys have
        value at least ``min_func_i``.
        """
        count_map = defaultdict(lambda: 0)
        if not argset:
            return count_map

        funcsets = [self.arg_to_funcset[arg] for arg in argset]
        # As an optimization below, we handle the largest funcset separately from
        # the others.
        largest_funcset = max(funcsets, key=len)

        for funcset in funcsets:
            if largest_funcset is funcset:
                continue
            for func_i in funcset:
                if func_i >= min_func_i:
                    count_map[func_i] += 1

        # We pick the smaller of the two containers (count_map, largest_funcset)
        # to iterate over to reduce the number of iterations needed.
        (smaller_funcs_container,
         larger_funcs_container) = sorted(
                 [largest_funcset, count_map],
                 key=len)

        for func_i in smaller_funcs_container:
            # Not already in count_map? It can't possibly be in the output, so
            # skip it.
            if count_map[func_i] < 1:
                continue

            if func_i in larger_funcs_container:
                count_map[func_i] += 1

        return {k: v for k, v in count_map.items() if v >= 2}

    def get_subset_candidates(self, argset, restrict_to_funcset=None):
        """
        Return a set of functions each of which whose argument list contains
        ``argset``, optionally filtered only to contain functions in
        ``restrict_to_funcset``.
        """
        iarg = iter(argset)

        indices = OrderedSet(
            fi for fi in self.arg_to_funcset[next(iarg)])

        if restrict_to_funcset is not None:
            indices &= restrict_to_funcset

        for arg in iarg:
            indices &= self.arg_to_funcset[arg]

        return indices

    def update_func_argset(self, func_i, new_argset):
        """
        Update a function with a new set of arguments.
        """
        new_args = OrderedSet(new_argset)
        old_args = self.func_to_argset[func_i]

        for deleted_arg in old_args - new_args:
            self.arg_to_funcset[deleted_arg].remove(func_i)
        for added_arg in new_args - old_args:
            self.arg_to_funcset[added_arg].add(func_i)

        self.func_to_argset[func_i].clear()
        self.func_to_argset[func_i].update(new_args)


if __name__ == '__main__':
    exps = [
        '-0.26*POW(MAX(fan_count_exp,1.0),-0.1262)*author_ct+1.0+-2',
        'POW(MAX(fan_count_exp,1.0),-0.1262)*author_ct+-2',
        '-0.26*POW(MAX(fan_count_exp,1.0),-0.1262)+-2'
    ]

    new = []
    for e in exps:
        tmp = ast.build_ast_all(e.replace(' ', ''))
        new.append(tmp)
    res = tree_cse(new, mod='math')
    print(res)


    exps = [
        '!a',
        "a && b && c",
        "a && c && b",
        "!a && b",
        "!(a || b) && c",
        "c && !(a || b) && a",
        "(a||b)&&(c||d)",
        "a&&b||c&&d"
    ]
    res = tree_cse([
        ast.build_ast_all(e.replace(' ', '')) for e in exps
    ])
    print(res)




