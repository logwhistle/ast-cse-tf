# -*- coding: utf-8 -*-
# contact: log_whistle@163.com
"""abstract syntax tree with only && || ( )"""

import tensorflow as tf
import numpy as np

VAR, FUNC, EOF = 'VAR', 'FUNC', 'EOF'
MULTIPLY, DEV, ADD, MINUS, LPAREN, RPAREN = '*', '/', '+', '-', '(', ')'
AND, OR, NOT, LPAREN, RPAREN = '&&', '||', '!', '(', ')'
SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5 = '==', '>', '>=', '<', '<='
COMPARE_SYMBOLS = ['=', '>', '<']

# RESERVED_SYMBOLS = {'*', '/', '+', '-', '(', ')'}
RESERVED_SYMBOLS = {'*', '/', '+', '-', '&', '|', '!', '(', ')', '=', '>', '<'}
MIN, MAX, SQRT, IF, POW, LOG = 'MIN', 'MAX', 'SQRT', 'IF', 'POW', 'LOG' 
WILSON, PWL3, PWL2, PCONT = 'WILSON_SCORE', 'PIECEWISE_LINEAR_3P', 'PIECEWISE_LINEAR_2P', 'PROPORTIONAL_CONTROL'
SPLIT = ','
DEFINEDOP = {} 
BOOLCACHE = set()


def build_ast_all(expression, binary_tree=False):
    root, _ = Parser(expression).parse()
    if binary_tree:
        return root
    else:
        return binary2ast(root).cut()


def binary2ast(root):
    children = []
    if root.value.type == VAR: 
        return Tree(root.value, children)
    if root.value.type == NOT:
        return Tree(root.value, [binary2ast(root.left)])
    if root.value.type == FUNC:  # FUNC是特殊的VAR，出现多次需要消除
        if root.value.value not in [PWL3, PWL2, PCONT]:
            children.append(root.left)
            if root.value.value == SQRT:
                return Tree(root.value, [binary2ast(child) for child in children])
            elif root.value.value == LOG:
                return Tree(root.value, [binary2ast(child) for child in children])
            elif root.value.value == IF:
                # return root
                children.append(root.right.left)
                children.append(root.right.right)
                tree = Tree(root.value, [binary2ast(child) for child in children])
                return tree
            children.append(root.right)
            return Tree(root.value, [binary2ast(child) for child in children])
        elif root.value.value == PCONT:
            children.append(root.left.left)
            children.append(root.left.right)
            children.append(root.right.left)
            children.append(root.right.right)
            return Tree(root.value, [binary2ast(child) for child in children])
        elif root.value.value == PWL2:
            children.append(root.left)
            children.append(root.right.left.left)
            children.append(root.right.left.right)
            children.append(root.right.right.left)
            children.append(root.right.right.right)
            return Tree(root.value, [binary2ast(child) for child in children])
        elif root.value.value == PWL2:
            children.append(root.left.left)
            children.append(root.left.right.left)
            children.append(root.left.right.right)
            children.append(root.right.left.left)
            children.append(root.right.left.right)
            children.append(root.right.right.left)
            children.append(root.right.right.right)
            return Tree(root.value, [binary2ast(child) for child in children])
    if root.value.type == MINUS:
        return Tree(root.value, [binary2ast(root.left)])
    root_type = root.value.type
    children.append(root.left)
    children.append(root.right)
    while True:
        new_children = []
        for child in children:
            if child.value.type == root_type:
                new_children.append(child.left)
                new_children.append(child.right)
            else:
                new_children.append(child)
        if len(new_children) == len(children):
            break
        children = new_children
    return Tree(root.value, [binary2ast(child) for child in children])

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError: 
        pass  
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

class Function_dictionary(object):
    def WHERE(self, cond, value1, value2):
        cond = tf.cast(cond, tf.float32)
        return cond * value1 + (1 - cond) * value2
    def WILSON_SCORE(self, p, n):
        score = tf.div(p + tf.div(2.0*const_value["ones"], n) - tf.div(1.0*const_value["ones"], n) * tf.sqrt(4.0 * n * (1.0 - p) * p + 4.0), 1.0 + tf.div(4.0*const_value["ones"], n))
        score = self.WHERE(tf.less_equal(n, 0.0) | tf.less(p, 0.0) | tf.greater(p, 1.0), const_value["zeros"], score)
        score = tf.clip_by_value(score, 0, 1)
        return score
    def PIECEWISE_LINEAR_2P(self, v, b0, b1, v0, v1):
        ans = self.WHERE(tf.greater_equal(v, b0)&tf.less(v ,b1), tf.div((v1 - v0), (b1 - b0)) * (v - b0) + v0, v1)
        ans = self.WHERE(tf.less(v, b0), v0, ans)
        return ans
    def PIECEWISE_LINEAR_3P(self, v, b0, b1, b2, v0, v1, v2):
        ans = self.WHERE(tf.greater_equal(v, b0)&tf.greater_equal(v, b1)&tf.less(v ,b2), tf.div((v2 - v1), (b2 - b1)) * (v - b1) + v1, v2)
        ans = self.WHERE(tf.greater_equal(v, b0)&tf.less(v ,b1), tf.div((v1 - v0), (b1 - b0)) * (v - b0) + v0, ans)
        ans = self.WHERE(tf.less(v, b0), v0, ans)
        return ans  
    def PROPORTIONAL_CONTROL(self, controlValue, targetValue, sec, secUpperBound):
        return self.WHERE(tf.greater_equal(controlValue, targetValue) | tf.greater_equal(sec, secUpperBound), const_value["zeros"], tf.div((targetValue - controlValue), (secUpperBound - sec)))


# print('xy_wilson', FuncDict.WILSON_SCORE(1.0, 2.0))

class Tree(object):
    def __init__(self, value, children):
        assert (value.type in [VAR, FUNC] and len(children) == 0) or (value.type == NOT and len(children) == 1)or (value.value == LOG and len(children) == 1)\
               or (value.type == MINUS and len(children) == 1) or (value.value == SQRT and len(children) == 1) or len(children) > 1
        self.op = value
        if self.op.value in [DEV, IF, POW, WILSON, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5, PCONT, PWL3, PWL2]:
            self.args = children
        else:
            self.args = tuple(sorted(children, key=lambda x: x.to_expression()))
        self._hash = None
        self._expression = None
        
        self.FuncDict = Function_dictionary()
        
        self.is_bool = False
        if self.op.value in [NOT, SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5, AND, OR]:
            self.is_bool = True

    def to_tfOp(self):
        if self.op.type == VAR:
            return globals()[self.op.value]
        name = 'TMP' + self.get_hash()
        if DEFINEDOP.get(name) != None:
            return DEFINEDOP[name]
        if self.op.type == FUNC:
            if self.op.value == MAX:
                tmp = tf.maximum(self.args[0].to_tfOp(), self.args[1].to_tfOp(), name)
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == MIN:
                tmp = tf.minimum(self.args[0].to_tfOp(), self.args[1].to_tfOp(), name)
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == POW:
                tmp = tf.pow(self.args[0].to_tfOp(), self.args[1].to_tfOp(), name)
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == WILSON:
                tmp = self.FuncDict.WILSON_SCORE(self.args[0].to_tfOp(), self.args[1].to_tfOp())
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == SQRT:
                tmp = tf.sqrt(self.args[0].to_tfOp(), name)
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == LOG:
                tmp_op = self.args[0].to_tfOp()
                cond = tf.not_equal(tmp_op, 0.0)
                tmp = tf.where(cond, tf.log(tmp_op, name), tf.zeros_like(tmp_op))
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == IF:
                cond = self.args[0].to_tfOp()
                arg_name = 'TMP' + self.args[0].get_hash()
                if (arg_name not in BOOLCACHE) and (self.args[0].op.value not in NAMEBOOLCACHE):
                    cond = tf.not_equal(cond, 0.0)
                # if tmp_op.dtype == tf.bool:
                #     cond = tf.not_equal(tmp_op, False)
                # elif tmp_op.dtype == tf.float32:
                #     cond = tf.not_equal(tmp_op, 0.0)
                # else:
                #     assert False
                tmp1, tmp2 = self.args[1].to_tfOp(), self.args[2].to_tfOp()
                arg_name1 = 'TMP' + self.args[1].get_hash()
                arg_name2 = 'TMP' + self.args[2].get_hash()
                if (arg_name1 in BOOLCACHE) or (self.args[1].op.value in NAMEBOOLCACHE):
                    tmp1 = tf.cast(tmp1, tf.float32)
                if (arg_name2 in BOOLCACHE) or (self.args[2].op.value in NAMEBOOLCACHE):
                    tmp2 = tf.cast(tmp2, tf.float32)
                tmp = self.FuncDict.WHERE(cond, tmp1, tmp2)
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == PCONT:
                tmp = self.FuncDict.PROPORTIONAL_CONTROL(self.args[0].to_tfOp(), self.args[1].to_tfOp(), self.args[2].to_tfOp(), self.args[3].to_tfOp())
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == PWL2:
                tmp = self.FuncDict.PIECEWISE_LINEAR_2P(self.args[0].to_tfOp(), self.args[1].to_tfOp(), self.args[2].to_tfOp(), self.args[3].to_tfOp(), self.args[4].to_tfOp())
                DEFINEDOP[name] = tmp
                return tmp
            elif self.op.value == PWL3:
                tmp = self.FuncDict.PIECEWISE_LINEAR_3P(self.args[0].to_tfOp(), self.args[1].to_tfOp(), self.args[2].to_tfOp(), self.args[3].to_tfOp(), self.args[4].to_tfOp(), self.args[5].to_tfOp(), self.args[6].to_tfOp())
                DEFINEDOP[name] = tmp
                return tmp
        
        
        if self.op.type == MINUS:
            tmp = -self.args[0].to_tfOp()
            DEFINEDOP[name] = tmp
            return tmp
        if self.op.type == NOT:
            tmp = self.args[0].to_tfOp()
            arg_name = 'TMP' + self.args[0].get_hash()
            # print('xy_debug', self.args[0], tmp.dtype, self.args[0].op.value, NAMEBOOLCACHE)
            if (arg_name not in BOOLCACHE) and (self.args[0].op.value not in NAMEBOOLCACHE):
                tmp = tf.not_equal(tmp, 0.0)
            # if tmp_op.dtype == tf.bool:
            #     tmp = ~tf.not_equal(tmp_op, False)
            # elif tmp_op.dtype == tf.float32:
            #     tmp = ~tf.not_equal(tmp_op, 0.0)
            # else:
            #     assert False
            DEFINEDOP[name] = ~tmp
            BOOLCACHE.add(name)
            self.is_bool = True
            return ~tmp
        if self.op.type in [SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5]:
            tmp1, tmp2 = self.args[0].to_tfOp(), self.args[1].to_tfOp()
            # tmp1, tmp2 = self.args[0].to_tfOp(), self.args[1].to_tfOp()
            if self.op.type == SLOGAN1:
                arg_name1, arg_name2 = 'TMP' + self.args[0].get_hash(), 'TMP' + self.args[1].get_hash()
                if (arg_name1 in BOOLCACHE) or (self.args[0].op.value in NAMEBOOLCACHE):
                    tmp1 = tf.cast(tmp1, tf.float32)
                if (arg_name2 in BOOLCACHE) or (self.args[1].op.value in NAMEBOOLCACHE):
                    tmp2 = tf.cast(tmp2, tf.float32)
                # tmp1, tmp2 = tf.cast(tmp1, tf.float32), tf.cast(tmp2, tf.float32)
                tmp = tf.equal(tmp1, tmp2, name)
                DEFINEDOP[name] = tmp
                BOOLCACHE.add(name)
                self.is_bool = True
                return tmp
            elif self.op.type == SLOGAN2:
                tmp = tf.greater(tmp1, tmp2, name)
                DEFINEDOP[name] = tmp
                BOOLCACHE.add(name)
                self.is_bool = True
                return tmp
            elif self.op.type == SLOGAN3:
                tmp = tf.greater_equal(tmp1, tmp2, name)
                DEFINEDOP[name] = tmp
                BOOLCACHE.add(name)
                self.is_bool = True
                return tmp
            elif self.op.type == SLOGAN4:
                tmp = tf.less(tmp1, tmp2, name)
                DEFINEDOP[name] = tmp
                BOOLCACHE.add(name)
                self.is_bool = True
                return tmp
            else:
                tmp = tf.less_equal(tmp1, tmp2, name)
                DEFINEDOP[name] = tmp
                BOOLCACHE.add(name)
                self.is_bool = True
                return tmp

        assert self.op.type in [MULTIPLY, ADD, DEV, AND, OR]
        if self.op.type == DEV:
            tmp = tf.div(self.args[0].to_tfOp(), self.args[1].to_tfOp(), name)
            DEFINEDOP[name] = tmp
            return tmp

        elif self.op.type == MULTIPLY:
            # flag = True
            # result = const_value["ones"]
            for i, arg in enumerate(self.args):
                if i == 0:
                    result = arg.to_tfOp()
                    # flag = False
                else:
                    tmp = arg.to_tfOp()
                    result = result * tmp
            DEFINEDOP[name] = result
            return result
        elif self.op.type == ADD:
            # flag = True
            # result = const_value["zeros"]
            for i, arg in enumerate(self.args):
                if i == 0:
                    result = arg.to_tfOp()
                    # flag = False
                else:
                    tmp = arg.to_tfOp()
                    result = result + tmp
            DEFINEDOP[name] = result
            return result
        elif self.op.type == AND:
            # flag = True
            for i, arg in enumerate(self.args):
                tmp = arg.to_tfOp()
                arg_name = 'TMP' + arg.get_hash()
                if (arg_name not in BOOLCACHE) and (arg.op.value not in NAMEBOOLCACHE):
                    tmp = tf.not_equal(tmp, 0.0)
                # if tmp_op.dtype == tf.bool:
                #     tmp = tf.not_equal(tmp_op, False)
                # elif tmp_op.dtype == tf.float32:
                #     tmp = tf.not_equal(tmp_op, 0.0)
                # else:
                #     assert False
                if i == 0:
                    result = tmp
                    # flag = False
                else:
                    result = result & tmp
            DEFINEDOP[name] = result
            BOOLCACHE.add(name)
            self.is_bool = True
            return result
        else:
            # flag = True
            for i, arg in enumerate(self.args):
                tmp = arg.to_tfOp()
                arg_name = 'TMP' + arg.get_hash()
                if (arg_name not in BOOLCACHE) and (arg.op.value not in NAMEBOOLCACHE):
                    tmp = tf.not_equal(tmp, 0.0)
                # if tmp_op.dtype == tf.bool:
                #     tmp = tf.not_equal(tmp_op, False)
                # elif tmp_op.dtype == tf.float32:
                #     tmp = tf.not_equal(tmp_op, 0.0)
                # else:
                #     assert False
                if i == 0:
                    result = tmp
                    # flag = False
                else:
                    result = result | tmp
            DEFINEDOP[name] = result
            BOOLCACHE.add(name)
            self.is_bool = True
            return result
        return self._tfOp
    
    
    def to_expression(self):
        if self.op.type == FUNC:
            if self.op.value == MAX:
                return MAX + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ')'
            elif self.op.value == MIN:
                return MIN + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ')'
            elif self.op.value == POW:
                return POW + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ')'
            elif self.op.value == WILSON:
                return WILSON + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ')'
            elif self.op.value == SQRT:
                return SQRT + '(' + self.args[0].to_expression() + ')'
            elif self.op.value == LOG:
                return LOG + '(' + self.args[0].to_expression() + ')'
            elif self.op.value == IF:
                return IF + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ',' + self.args[2].to_expression() + ')'
            elif self.op.value == PCONT:
                return PCONT + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ',' + self.args[2].to_expression() + ',' + self.args[3].to_expression() +  ')'
            elif self.op.value == PWL2:
                return PWL2 + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ',' + self.args[2].to_expression() + ',' + self.args[3].to_expression() + ',' + self.args[4].to_expression() +  ')'
            elif self.op.value == PWL3:
                return PWL3 + '(' + self.args[0].to_expression() + ',' + self.args[1].to_expression() + ',' + self.args[2].to_expression() + ',' + self.args[3].to_expression() + ',' + self.args[4].to_expression() + ',' + self.args[5].to_expression() + ',' + self.args[6].to_expression() +  ')'
        if self._expression is not None:
            return self._expression
        if self.op.type == VAR:
            return self.op.value
        if self.op.type == MINUS:
            if self.args[0].op.type == VAR:
                return MINUS + self.args[0].to_expression()
            else:
                return MINUS + '(' + self.args[0].to_expression() + ')'
        if self.op.type == NOT:
            if self.args[0].op.type == VAR:
                return NOT + self.args[0].to_expression()
            else:
                return NOT + '(' + self.args[0].to_expression() + ')'
        if self.op.type in [SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5]:
            tmp0, tmp1 = self.args[0].to_expression(), self.args[1].to_expression()
            if '+' in tmp0 or '*' in tmp0:
                tmp0 = '(' + self.args[0].to_expression() + ')'
            if '+' in tmp1 or '*' in tmp1:
                tmp1 = '(' + self.args[1].to_expression() + ')'
            # return '(' + tmp0 + self.op.type + tmp1 + ')'
            return tmp0 + self.op.type + tmp1
        assert self.op.type in [MULTIPLY, ADD, DEV, AND, OR]
        if self.op.type == DEV:
            tmp0, tmp1 = self.args[0].to_expression(), self.args[1].to_expression()
            if '+' in tmp0 or '*' in tmp0:
                tmp0 = '(' + self.args[0].to_expression() + ')'
            if '+' in tmp1 or '*' in tmp1:
                tmp1 = '(' + self.args[1].to_expression() + ')'
            return tmp0 + self.op.type + tmp1
        elif self.op.type == MULTIPLY:
            result = ''
            for arg in self.args:
                if arg.op.type == ADD:
                    result += '(' + arg.to_expression() + ')'
                else:
                    result += arg.to_expression()
                result += MULTIPLY
            self._expression = result[:-1]
        elif self.op.type == ADD:
            self._expression = ADD.join(arg.to_expression() for arg in self.args)
        elif self.op.type == AND:
            result = ''
            for arg in self.args:
                if arg.op.type == OR:
                    result += '(' + arg.to_expression() + ')'
                else:
                    result += arg.to_expression()
                result += AND
            self._expression = result[:-2]
        else:
            self._expression = OR.join(arg.to_expression() for arg in self.args)
        return self._expression

    def visit(self, prefix=''):
        print('%s%s' % (prefix, self.op))
        for arg in self.args:
            arg.visit(prefix + '-')

    def cut(self):
        if self.op.type in [SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5]:
            return self
        if self.op.type == DEV:
            return self
        if self.op.type == VAR:
            return Tree(self.op, [])
        if self.op.type == FUNC:
            if self.op.value in [MAX, MIN, WILSON, POW]:
                newarg0 = self.args[0].cut()
                newarg1 = self.args[1].cut()
                return Tree(self.op, [newarg0, newarg1])
            elif self.op.value in [SQRT, LOG]:
                return Tree(self.op, [self.args[0].cut()])
            elif self.op.value in [IF]:
                newarg0 = self.args[0].cut()
                newarg1 = self.args[1].cut()
                newarg2 = self.args[2].cut()
                return Tree(self.op, [newarg0, newarg1, newarg2])
            elif self.op.value in [PCONT]:
                newarg0 = self.args[0].cut()
                newarg1 = self.args[1].cut()
                newarg2 = self.args[2].cut()
                newarg3 = self.args[3].cut()
                return Tree(self.op, [newarg0, newarg1, newarg2, newarg3])
            elif self.op.value in [PWL2]:
                newarg0 = self.args[0].cut()
                newarg1 = self.args[1].cut()
                newarg2 = self.args[2].cut()
                newarg3 = self.args[3].cut()
                newarg4 = self.args[4].cut()
                return Tree(self.op, [newarg0, newarg1, newarg2, newarg3, newarg4])
            elif self.op.value in [PWL3]:
                newarg0 = self.args[0].cut()
                newarg1 = self.args[1].cut()
                newarg2 = self.args[2].cut()
                newarg3 = self.args[3].cut()
                newarg4 = self.args[4].cut()
                newarg5 = self.args[5].cut()
                newarg6 = self.args[6].cut()
                return Tree(self.op, [newarg0, newarg1, newarg2, newarg3, newarg4, newarg5, newarg6])
            else:
                return self
        if self.op.type == NOT:
            if self.args[0].op.value == 'False' or self.args[0].op.value == '0' or self.args[0].op.value == '0.0':
                new_value = Token(VAR, 'True')
                return Tree(new_value, [])
            elif self.args[0].op.value == 'True' or self.args[0].op.value == '1' or self.args[0].op.value == '1.0':
                new_value = Token(VAR, 'False')
                return Tree(new_value, [])
            else:
                tmp = self.args[0].cut()
                if tmp.op.value == 'False' or tmp.op.value == '0' or tmp.op.value == '0.0':
                    new_value = Token(VAR, 'True')
                    return Tree(new_value, [])
                elif tmp.op.value == 'True' or tmp.op.value == '1' or tmp.op.value == '1.0':
                    new_value = Token(VAR, 'False')
                    return Tree(new_value, [])
                else:
                    return Tree(self.op, [tmp])
        if self.op.type == MINUS:
            if self.args[0].op.value == '0' or self.args[0].op.value == '0.0':
                new_value = Token(VAR, '0')
                return Tree(new_value, [])
            else:
                tmp = self.args[0].cut()
                if tmp.op.value == '0' or tmp.op.value == '0.0':
                    new_value = Token(VAR, '0')
                    return Tree(new_value, [])
                else:
                    return Tree(self.op, [tmp])
        if self.op.type == MULTIPLY:
            new_children = []
            for arg in self.args:
                if arg.op.type == VAR:
                    if arg.op.value == '0' or arg.op.value == '0.0':
                        new_value = Token(VAR, '0')
                        # self.op = new_value
                        # self.args = []
                        # return Tree(self.op, self.args)
                        return Tree(new_value, [])
                    else:
                        new_children.append(arg)
                elif arg.op.type == FUNC:
                    new_children.append(arg)
                else:
                    tmp = arg.cut()
                    if tmp.op.value != '0' and tmp.op.value != '0.0':
                        new_children.append(tmp)
            if len(new_children) == 1:
                return new_children[0]
            else:
                new_children0 = tuple(sorted(new_children, key=lambda x: x.to_expression()))
                return Tree(self.op, new_children0)
        if self.op.type == ADD:
            new_children = []
            for arg in self.args:
                if arg.op.type == VAR:
                    if arg.op.value == '0' or arg.op.value == '0.0':
                        continue
                    else:
                        new_children.append(arg)
                elif arg.op.type == FUNC:
                    new_children.append(arg)
                else:
                    tmp = arg.cut()
                    if tmp.op.value != '0' and tmp.op.value != '0.0':
                        new_children.append(tmp)
            if new_children == []:
                new_value = Token(VAR, '0')
                return Tree(new_value, [])
            elif len(new_children) == 1:
                return new_children[0]
            else:
                new_children0 = tuple(sorted(new_children, key=lambda x: x.to_expression()))
                return Tree(self.op, new_children0)
        if self.op.type == AND:
            new_children = []
            for arg in self.args:
                if arg.op.type == VAR:
                    if arg.op.value == 'False' or arg.op.value == '0' or arg.op.value == '0.0':
                        new_value = Token(VAR, 'False')
                        return Tree(new_value, [])
                    elif arg.op.value == 'True' or arg.op.value == '1' or arg.op.value == '1.0':
                        continue
                    else:
                        new_children.append(arg)
                elif arg.op.type == FUNC:
                    new_children.append(arg)
                else:
                    tmp = arg.cut()
                    if tmp.op.value == 'False' or tmp.op.value == '0' or tmp.op.value == '0.0':
                        new_value = Token(VAR, 'False')
                        return Tree(new_value, [])
                    elif tmp.op.value == 'True' or tmp.op.value == '1' or tmp.op.value == '1.0':
                        continue
                    else:
                        new_children.append(tmp)
            if new_children == []:
                new_value = Token(VAR, 'True')
                return Tree(new_value, [])
            elif len(new_children) == 1:
                return new_children[0]
            else:
                new_children0 = tuple(sorted(new_children, key=lambda x: x.to_expression()))
                return Tree(self.op, new_children0)
        if self.op.type == OR:
            new_children = []
            for arg in self.args:
                if arg.op.type == VAR:
                    if arg.op.value == 'False' or arg.op.value == '0' or arg.op.value == '0.0':
                        continue
                    elif arg.op.value == 'True' or arg.op.value == '1' or arg.op.value == '1.0':
                        new_value = Token(VAR, 'True')
                        return Tree(new_value, [])
                    else:
                        new_children.append(arg)
                elif arg.op.type == FUNC:
                    new_children.append(arg)
                else:
                    tmp = arg.cut()
                    if tmp.op.value == 'False' or tmp.op.value == '0' or tmp.op.value == '0.0':
                        continue
                    elif tmp.op.value == 'True' or tmp.op.value == '1' or tmp.op.value == '1.0':
                        new_value = Token(VAR, 'True')
                        return Tree(new_value, [])
                    else:
                        new_children.append(tmp)
                    
            if new_children == []:
                new_value = Token(VAR, 'False')
                return Tree(new_value, [])
            elif len(new_children) == 1:
                return new_children[0]
            else:
                new_children0 = tuple(sorted(new_children, key=lambda x: x.to_expression()))
                return Tree(self.op, new_children0)


    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.op.type, self.op.value) + tuple(self.args))
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Tree):
            return (self.op == other.op) and (self.args == other.args)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.to_expression()

    def __repr__(self):
        return self.__str__()

    def get_hash(self):
        return str(self.__hash__())

class BinaryTree(object):
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

    def to_expression(self):
        if self.value.type == FUNC:
            if self.value.value == MAX:
                return MAX + '(' + self.left.to_expression() + ',' + self.right.to_expression() + ')'
            elif self.value.value == MIN:
                return MIN + '(' + self.left.to_expression() + ',' + self.right.to_expression() + ')'
            elif self.value.value == POW:
                return POW + '(' + self.left.to_expression() + ',' + self.right.to_expression() + ')'
            elif self.value.value == WILSON:
                return WILSON + '(' + self.left.to_expression() + ',' + self.right.to_expression() + ')'
            elif self.value.value == SQRT:
                if self.left.value.type == VAR:
                    return SQRT + self.left.to_expression()
                else:
                    return SQRT + '(' + self.left.to_expression() + ')'
            elif self.value.value == LOG:
                if self.left.value.type == VAR:
                    return LOG + self.left.to_expression()
                else:
                    return LOG + '(' + self.left.to_expression() + ')'
            elif self.value.value == IF:
                return IF + '(' + self.left.to_expression() + ',' + self.right.left.to_expression() + ',' + self.right.right.to_expression() + ')'
            elif self.value.value == PCONT:
                return PCONT + '(' + self.left.left.to_expression() + ',' + self.left.right.to_expression() + ',' + self.right.left.to_expression() + ',' + self.right.right.to_expression() + ')'
            elif self.value.value == PWL2:
                return PWL2 + '(' + self.left.to_expression() + ',' + self.right.left.left.to_expression() + ',' + self.right.left.right.to_expression() + ',' + self.right.right.left.to_expression() + ',' + self.right.right.right.to_expression() + ')'
            elif self.value.value == PWL3:
                return PWL3 + '(' + self.left.left.to_expression() + ',' + self.left.right.left.to_expression() + ',' + self.left.right.right.to_expression() + ',' + self.right.left.left.to_expression() + ',' + self.right.left.right.to_expression() + ',' + self.right.right.left.to_expression() + ',' + self.right.right.right.to_expression() + ')'
        if self.value.type == VAR:
            return self.value.value
        if self.value.type == MINUS:
            if self.left.value.type == VAR:
                return MINUS + self.left.to_expression()
            else:
                return MINUS + '(' + self.left.to_expression() + ')'
        if self.value.type == NOT:
            if self.left.value.type == VAR:
                return NOT + self.left.to_expression()
            else:
                return NOT + '(' + self.left.to_expression() + ')'
        if self.value.type in [SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5]:
            tmp0, tmp1 = self.left.to_expression(), self.right.to_expression()
            if '+' in tmp0 or '*' in tmp0:
                tmp0 = '(' + self.left.to_expression() + ')'
            if '+' in tmp1 or '*' in tmp1:
                tmp1 = '(' + self.right.to_expression() + ')'
            # return '(' + tmp0 + self.value.type + tmp1 + ')'
            return tmp0 + self.value.type + tmp1
        assert self.value.type in [MULTIPLY, ADD, DEV, AND, OR]
        if self.value.type == DEV:
            tmp0, tmp1 = self.left.to_expression(), self.right.to_expression()
            if '+' in tmp0 or '*' in tmp0:
                tmp0 = '(' + self.left.to_expression() + ')'
            if '+' in tmp1 or '*' in tmp1:
                tmp1 = '(' + self.right.to_expression() + ')'
            return tmp0 + self.value.type + tmp1
        if self.value.type == MULTIPLY and self.left.value.type == ADD:
            left = '(' + self.left.to_expression() + ')'
        else:
            left = self.left.to_expression()
        if self.value.type == MULTIPLY and self.right.value.type == ADD:
            right = '(' + self.right.to_expression() + ')'
        else:
            right = self.right.to_expression()
        return left + self.value.value + right

    def visit(self):
        print(self.value)
        print('left tree')
        print(None if self.left is None else self.left.visit())
        print('right tree')
        print(None if self.right is None else self.right.visit())

    def __str__(self):
        return self.to_expression()

    def __repr__(self):
        return self.__str__()


class Token(object):
    def __init__(self, vtype, value=None):
        self.type = vtype
        self.value = value if vtype in [VAR, FUNC] else vtype
        if vtype == VAR:
            assert all(symbol not in value for symbol in RESERVED_SYMBOLS)

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Token):
            return (self.type == other.type) and (self.value == other.value)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Lexer(object):
    def __init__(self, text, func_flag=False):
        self.text = text
        self.position = 0
        self.current_char = self.text[self.position]
        self.func_flag = func_flag
        if func_flag:
            RESERVED_SYMBOLS.add(',')
        self.RESERVED_SYMBOLS_FUNC = RESERVED_SYMBOLS

    def advance(self, step=1):
        self.position += step
        if self.position >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.position]

    def get_variable(self):
        n_lparen = 0
        is_func = False
        result = ''
        while True:
            result += self.current_char
            self.advance()
            if self.current_char is None:
                break
            if self.current_char == LPAREN:  # 发现func
                is_func = True
                n_lparen += 1
            elif n_lparen > 0 and self.current_char == RPAREN:   # func内部，stack去除右括号
                n_lparen -= 1
            elif n_lparen == 0 and self.current_char in self.RESERVED_SYMBOLS_FUNC:  # 非func内部，出现保留字
                break
            
        return result, is_func

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char not in self.RESERVED_SYMBOLS_FUNC:
                value, is_func = self.get_variable()
                return Token(FUNC if is_func else VAR, value), self.position
            if self.current_char == '!':
                self.advance()
                return Token(NOT), self.position
            if self.current_char == '&':
                self.advance(2)
                return Token(AND), self.position
            if self.current_char == '|':
                self.advance(2)
                return Token(OR), self.position
            if self.current_char == '-':
                self.advance()
                return Token(MINUS), self.position
            if self.current_char == '*':
                self.advance()
                return Token(MULTIPLY), self.position
            if self.current_char == '/':
                self.advance()
                return Token(DEV), self.position
            if self.current_char == '+':
                self.advance()
                return Token(ADD), self.position
            if self.current_char == '(':
                self.advance()
                return Token(LPAREN), self.position
            if self.current_char == ')':
                self.advance()
                return Token(RPAREN), self.position
            if self.current_char in COMPARE_SYMBOLS:
                if self.current_char == '=':
                    self.advance(2)
                    return Token(SLOGAN1), self.position
                elif self.current_char == '>':
                    self.advance()
                    if self.current_char == '=':
                        self.advance()
                        return Token(SLOGAN3), self.position
                    else:
                        return Token(SLOGAN2), self.position
                elif self.current_char == '<':
                    self.advance()
                    if self.current_char == '=':
                        self.advance()
                        return Token(SLOGAN5), self.position
                    else:
                        return Token(SLOGAN4), self.position
            if self.func_flag and self.current_char == ',':
                self.advance()
                return Token(SPLIT), self.position
        return Token(EOF), self.position

    def split(self):
        result = []
        while True:
            token = self.get_next_token()
            result.append(token)
            if token.type == EOF:
                break
        return result


class Parser(object):
    def __init__(self, text, func_flag=False):
        self.lexer = Lexer(text, func_flag)
        self.current_token, self.position = self.lexer.get_next_token()

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token, self.position = self.lexer.get_next_token()
        else:
            raise ValueError

    def factor(self):
        token = self.current_token
        if token.type == VAR:
            if (flag_dict['raw_variable'].get(token.value) is None) and (flag_dict['defined_variable'].get(token.value) is None):
                with tf.variable_scope("undefined_variable", reuse=tf.AUTO_REUSE):
                    if is_number(token.value):
                        globals()[token.value] = tf.cast(eval(token.value), tf.float32)*const_value["ones"]
                    else:
                        # globals()[token.value] = tf.get_variable(token.value, initializer=0.0)
                        if all_defined.get(token.value) is None:
                            print('undefined_variable_in_parse: ', token.value)
                            globals()[token.value] = const_value["zeros"]
                        else:
                            define_method = add_minux(all_defined[token.value].replace(' ', ''))
                            define_method = build_ast_all(define_method)
                            flag_dict['defined_variable'][token.value] = define_method.to_tfOp()
                            if define_method.is_bool:
                                NAMEBOOLCACHE.add(token.value)
                            globals()[token.value] = flag_dict['defined_variable'][token.value]
                            
            else:
                if flag_dict['raw_variable'].get(token.value) is None:
                    globals()[token.value] = flag_dict['defined_variable'].get(token.value)
                else:
                    globals()[token.value] = flag_dict['raw_variable'].get(token.value)
                
            self.eat(token.type)
            return BinaryTree(token, None, None), self.position
        elif token.type == FUNC:
            tree = self.parse_func(token)
            self.eat(token.type)
            return tree, self.position
        elif token.type == MINUS:
            self.eat(MINUS)
            tree0, self.position = self.factor()
            return BinaryTree(token, tree0, None), self.position
        elif token.type == NOT:
            self.eat(NOT)
            tree0, self.position = self.factor()
            # print('parse NOT: ', tree0)
            return BinaryTree(token, tree0, None), self.position
        elif token.type == LPAREN:
            self.eat(LPAREN)
            tree, self.position = self.parse()
            self.eat(RPAREN)
            return tree, self.position
        else:
            print('error token', token)
            raise ValueError

    def dev(self):
        # print('factor token before: ', self.current_token)
        tree, self.position = self.factor()
        # print('factor token after: ', self.current_token)
        while self.current_token.type == DEV:
            token = self.current_token
            self.eat(DEV)
            tree1, self.position = self.factor()
            tree = BinaryTree(token, tree, tree1)
        return tree, self.position
    
    def mul(self):
        tree, self.position = self.dev()
        while self.current_token.type == MULTIPLY:
            token = self.current_token
            self.eat(MULTIPLY)
            tree1, self.position = self.dev()
            tree = BinaryTree(token, tree, tree1)
        return tree, self.position

    def add(self):
        tree, self.position = self.mul()
        while self.current_token.type == ADD:
            token = self.current_token
            self.eat(ADD)
            tree1, self.position = self.mul()
            tree = BinaryTree(token, tree, tree1)
        return tree, self.position
    
    def compare(self):
        tree, self.position = self.add()
        while self.current_token.type in [SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5]:
            token = self.current_token
            self.eat(self.current_token.type)
            tree1, self.position = self.add()
            tree = BinaryTree(token, tree, tree1)
        return tree, self.position

    def term(self):
        tree, self.position = self.compare()
        while self.current_token.type == AND:
            token = self.current_token
            self.eat(AND)
            tree1, self.position = self.compare()
            tree = BinaryTree(token, tree, tree1)
        return tree, self.position

    def parse(self):
        tree, self.position = self.term()
        while self.current_token.type == OR:
            token = self.current_token
            self.eat(OR)
            tree1, self.position = self.term()
            tree = BinaryTree(token, tree, tree1)
        return tree, self.position
    

    def parse_func(self, token):
        current_text = token.value
        if len(current_text) > 3 and (current_text[:3] == MAX):
            if current_text[:4] == 'MAX(' and current_text[-1] == ')':
                current_text = current_text[4:-1]
            else:
                raise ValueError
            tree0, position = Parser(current_text, func_flag=True).parse()
            # eat_text = tree0.to_expression()
            # current_text = current_text[len(eat_text)+1:]
            if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                raise ValueError
            current_text = current_text[position:]
            tree1, position = Parser(current_text, func_flag=True).parse()
            tree = BinaryTree(Token(FUNC, MAX), tree0, tree1)
            return tree
        elif len(current_text) > 3 and (current_text[:3] == MIN):
            if current_text[:4] == 'MIN(' and current_text[-1] == ')':
                current_text = current_text[4:-1]
            else:
                raise ValueError
            tree0, position = Parser(current_text, func_flag=True).parse()
            # eat_text = tree0.to_expression()
            if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                raise ValueError
            # current_text = current_text[len(eat_text)+1:]
            current_text = current_text[position:]
            tree1, position = Parser(current_text, func_flag=True).parse()
            tree = BinaryTree(Token(FUNC, MIN), tree0, tree1)
            return tree
        elif len(current_text) > 3 and (current_text[:3] == POW):
            if current_text[:4] == 'POW(' and current_text[-1] == ')':
                current_text = current_text[4:-1]
            else:
                raise ValueError
            tree0, position = Parser(current_text, func_flag=True).parse()
            # eat_text = tree0.to_expression()
            if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                raise ValueError
            # current_text = current_text[len(eat_text)+1:]
            current_text = current_text[position:]
            tree1, position = Parser(current_text, func_flag=True).parse()
            tree = BinaryTree(Token(FUNC, POW), tree0, tree1)
            return tree
        elif len(current_text) > 12 and (current_text[:12] == WILSON):
            if current_text[:13] == 'WILSON_SCORE(' and current_text[-1] == ')':
                current_text = current_text[13:-1]
            else:
                raise ValueError
            tree0, position = Parser(current_text, func_flag=True).parse()
            # eat_text = tree0.to_expression()
            if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                raise ValueError
            # current_text = current_text[len(eat_text)+1:]
            current_text = current_text[position:]
            tree1, position = Parser(current_text, func_flag=True).parse()
            tree = BinaryTree(Token(FUNC, WILSON), tree0, tree1)
            return tree
        elif len(current_text) > 3 and (current_text[:3] == LOG):
            if current_text[:4] == 'LOG(' and current_text[-1] == ')':
                current_text = current_text[4:-1]
            else:
                raise ValueError
            tree0, position = Parser(current_text, func_flag=True).parse()
            tree = BinaryTree(Token(FUNC, LOG), tree0, None)
            return tree
        elif len(current_text) > 4 and (current_text[:4] == SQRT):
            if current_text[:5] == 'SQRT(' and current_text[-1] == ')':
                current_text = current_text[5:-1]
            else:
                raise ValueError
            tree0, position = Parser(current_text, func_flag=True).parse()
            tree = BinaryTree(Token(FUNC, SQRT), tree0, None)
            return tree
        elif len(current_text) > 2 and (current_text[:2] == IF):
            if current_text[:3] == 'IF(' and current_text[-1] == ')':
                current_text = current_text[3:-1]
            else:
                raise ValueError
            # print('before_tree0', current_text)
            tree0, position = Parser(current_text, func_flag=True).parse()
            # print('tree0', tree0)
            if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                raise ValueError

            # print('tree0_context', current_text)
            current_text = current_text[position:]
            # current_text = current_text[len(eat_text0)+1:]
            # print('tree1_context', current_text)
            tree1, position = Parser(current_text, func_flag=True).parse()

            # eat_text1 = tree1.to_expression()
            if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                raise ValueError
            # current_text = current_text[len(eat_text1)+1:]
            current_text = current_text[position:]
            tree2, position = Parser(current_text, func_flag=True).parse()
            tree_tmp = BinaryTree(Token(FUNC, 'TMP'), tree1, tree2)
            tree = BinaryTree(Token(FUNC, IF), tree0, tree_tmp)
            # tree = Tree(Token(FUNC, IF), binary2ast(child for child in [tree0, tree1, tree2]))
            return tree
        elif len(current_text) > 20 and (current_text[:20] == PCONT):
            if current_text[:21] == 'PROPORTIONAL_CONTROL(' and current_text[-1] == ')':
                current_text = current_text[21:-1]
            else:
                raise ValueError
            # print('before_tree0', current_text)
            trees = []
            for num in range(4):
                tree0, position = Parser(current_text, func_flag=True).parse()
                trees.append(tree0)
                # print('tree0', tree0)
                if num < 3:
                    if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                        raise ValueError
                    current_text = current_text[position:]
                    
            tree_tmp0 = BinaryTree(Token(FUNC, 'TMP0'), trees[0], trees[1])
            tree_tmp1 = BinaryTree(Token(FUNC, 'TMP1'), trees[2], trees[3])
            tree = BinaryTree(Token(FUNC, PCONT), tree_tmp0, tree_tmp1)
            return tree
        elif len(current_text) > 19 and (current_text[:19] == PWL2):
            if current_text[:20] == 'PIECEWISE_LINEAR_2P(' and current_text[-1] == ')':
                current_text = current_text[20:-1]
            else:
                raise ValueError
            # print('before_tree0', current_text)
            trees = []
            for num in range(5):
                tree0, position = Parser(current_text, func_flag=True).parse()
                trees.append(tree0)
                # print('tree0', tree0)
                if num < 4:
                    if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                        raise ValueError
                    current_text = current_text[position:]
                    
            tree_tmp0 = BinaryTree(Token(FUNC, 'TMP0'), trees[1], trees[2])
            tree_tmp1 = BinaryTree(Token(FUNC, 'TMP1'), trees[3], trees[4])
            tree_tmp2 = BinaryTree(Token(FUNC, 'TMP2'), tree_tmp0, tree_tmp1)
            tree = BinaryTree(Token(FUNC, PWL2), trees[0], tree_tmp2)
            return tree
        elif len(current_text) > 19 and (current_text[:19] == PWL3):
            if current_text[:20] == 'PIECEWISE_LINEAR_3P(' and current_text[-1] == ')':
                current_text = current_text[20:-1]
            else:
                raise ValueError
            # print('before_tree0', current_text)
            trees = []
            for num in range(7):
                tree0, position = Parser(current_text, func_flag=True).parse()
                trees.append(tree0)
                # print('tree0', tree0)
                if num < 6:
                    if position-1 < len(current_text) and (current_text[position-1] != ',' or current_text[position] == ','):
                        raise ValueError
                    current_text = current_text[position:]
                    
            tree_tmp0 = BinaryTree(Token(FUNC, 'TMP0'), trees[1], trees[2])
            tree_tmp1 = BinaryTree(Token(FUNC, 'TMP1'), trees[3], trees[4])
            tree_tmp2 = BinaryTree(Token(FUNC, 'TMP2'), trees[5], trees[6])
            tree_tmp3 = BinaryTree(Token(FUNC, 'TMP3'), trees[0], tree_tmp0)
            tree_tmp4 = BinaryTree(Token(FUNC, 'TMP4'), tree_tmp1, tree_tmp2)
            tree = BinaryTree(Token(FUNC, PWL3), tree_tmp3, tree_tmp4)
            return tree
        else:
            print('please contact xunyou to add new function')
            raise ValueError
            # token = Token(VAR, token.value)
            # return BinaryTree(token, None, None)


def add_minux(curr_text):
    newchar0, curr_text = "", str(curr_text).replace(' ', '')
    for idx, char in enumerate(curr_text):
        if char == "-" and idx > 0 and curr_text[idx-1] not in ['(', ',', '<', '=', '>', '&', '|', '+', '*', '/', '!']:
            newchar0 += "+"
        newchar0 += char
    return newchar0

def cut_minux(curr_text):
    newchar0, curr_text = "", str(curr_text).replace(' ', '')
    for idx, char in enumerate(curr_text):
        if char == "+" and idx+1 < len(curr_text) and curr_text[idx+1] == '-':
            char0 = ''
        else:
            char0 = char
        newchar0 += char0
    return newchar0   


if __name__ == '__main__':
    # test value model
    import json
    import copy
    heads, heads_base, heads_mul = set(), {}, {}
    get_note_flag, note_var = True, {}
    inputs, outputs = {}, {}  # for input and output signature_def
    flag_dict = {'raw_variable':{}, 'defined_variable':{}}
    NAMEBOOLCACHE = set()
    # test example
    ins_prt = {"requestFeatures":{'r1':['float'], 'r2':['int'], 'r3':['double']}, "noteFeatures":{'n1':['float'], 'n2':['int']}, "defined_features":{'d1':'n2/1.2 + MAX(0.0, 0.08 * (0.11 - r1 + n1) / 0.11)', 'd2':'IF(r3*2>1&&(n2==10), -r1*n2/n1*r2+MAX(MIN(r1,r2),WILSON_SCORE(r3, n1))*SQRT(WILSON_SCORE(n2, r1))-2, r2)'}}
    data = {
        'base':{
            '(r1==2)|(n2>10)&&(MIN(2, d2)<3)':{'click':'2', 'dwell-time':'2*r1+d1', 'like':'n1+r3', 'hide':'r2-n1'},
            '!(r2==5)&&(d1<10)':{'click':'3', 'dwell-time':'2*r3+d2', 'like':'r2', 'hide':'r1-n2'}
        },
        'multipliers':{
            '(r3>=2)|(n2<=0)&&!(MIN(2, d1)>5)':{'click':'1.5', 'dwell-time':'2-d1', 'like':'n1', 'hide':'r2'},
            '!(d1>5)':{'click':'1.2', 'dwell-time':'2.1', 'like':'n2', 'hide':'r1'}
        }
    }

    req_nums, note_nums = 0, 0

        
    requestFeatures = tf.placeholder(shape=[None], dtype=tf.float32)
    noteFeatures = tf.placeholder(shape=[None, None], dtype=tf.float32)
    inputs["requestFeatures"] = tf.saved_model.utils.build_tensor_info(requestFeatures)
    inputs["noteFeatures"] = tf.saved_model.utils.build_tensor_info(noteFeatures)
    for i, req in enumerate(sorted(ins_prt["requestFeatures"].keys())):
        flag_dict['raw_variable'][req] = requestFeatures[i]
        req_nums += 1
    for i, note in enumerate(sorted(ins_prt["noteFeatures"].keys())):
        flag_dict['raw_variable'][note] = noteFeatures[i, :]
        note_nums += 1
        if get_note_flag:
            # request features batch_size=1, note = n   set batch_size here
            note_var['batch_size'] = flag_dict['raw_variable'][note]
            get_note_flag = False

    const_value = {}
    with tf.variable_scope("const_value", reuse=tf.AUTO_REUSE):
        const_value["zeros"] = tf.zeros_like(note_var['batch_size'], dtype=tf.float32)
        const_value["ones"] = tf.ones_like(note_var['batch_size'], dtype=tf.float32)
    
    
    # all tasks in MTL
    heads = ['click', 'dwell-time', 'like', 'hide']
    for item in heads:
        heads_base[item] = tf.zeros_like(note_var['batch_size'], dtype=tf.float32, name='base_'+item)
        heads_mul[item] = tf.ones_like(note_var['batch_size'], dtype=tf.float32, name='multipliers_'+item)


    tmp_dict = sorted(ins_prt['defined_features'].items(), key=lambda x:x[0])
    all_defined = {}
    for define_var, define_method in tmp_dict:
        all_defined[define_var] = define_method

    with tf.variable_scope("defined_variable", reuse=tf.AUTO_REUSE):
        for define_var, define_method in tmp_dict:
            if flag_dict['defined_variable'].get(define_var) is None:
                define_method = add_minux(define_method.replace(' ', ''))
                define_method = build_ast_all(define_method)
                flag_dict['defined_variable'][define_var] = define_method.to_tfOp()
                if define_method.is_bool:
                    NAMEBOOLCACHE.add(define_var)
    
    # # 图内打印
    # sess0 = tf.Session()
    # with open('/Users/user/code/test/deleteVM_sim/v5/test_curl_new.txt', 'r', encoding='utf-8') as f:
    #     content = f.read().split('\n')
    #     input = {}
    #     for item0 in content:
    #         item = item0[88:-1]
    #         # print(item)
    #         name, value = item.split(' [')
    #         if name == 'LAMBDA_REQUEST_CONTEXT':
    #             continue
    #         values = list(map(float, value.split(' ')))
    #         input[flag_dict['raw_variable'][name]] = values
    
    vm = {'base': dict(), 'multipliers': dict()}
    for key0, value0 in data['base'].items():
        if str(key0) == 'true':
            for key1, expression in value0.items():
                expression = str(expression)
                expression = add_minux(expression.replace(' ', ''))
                simplify_value1 = build_ast_all(expression)
                heads_base[key1] += simplify_value1.to_tfOp()

        else:
            key0 = add_minux(key0.replace(' ', ''))
            simplify_key0 = build_ast_all(key0)
            simplify_key0 = tf.cast(simplify_key0.to_tfOp(), tf.float32)

            for key1, expression in value0.items():
                expression = str(expression)
                expression = add_minux(expression.replace(' ', ''))
                simplify_value1 = build_ast_all(expression)
                heads_base[key1] += simplify_key0 * simplify_value1.to_tfOp()

                # 图内细节  
                # if key1 == 'hide':
                #     key11 = key0 + " + " + expression
                #     tmp = simplify_key0 * simplify_value1.to_tfOp()
                #     # value = sess0.run([simplify_key0, simplify_value1.to_tfOp()], feed_dict=input)
                #     value = sess0.run(tmp, feed_dict=input)
                #     print(key11, " :  ", value)

                # heads_base[key1] += tf.where(simplify_key0, simplify_value1.to_tfOp(), tf.zeros_like(note_var['batch_size'], dtype=tf.float32)) 
    print('done2', '=================================information========================================')

    print(heads_base)

    # for key0, value0 in data['multipliers'].items():
    for key1, value1 in data['multipliers'].items():
        key1 = add_minux(key1.replace(' ', ''))
        simplify_key1 = build_ast_all(key1)
        # simplify_key1 = tf.cast(tf.cast(simplify_key1.to_tfOp(), tf.bool), tf.float32)
        simplify_key1 = tf.cast(simplify_key1.to_tfOp(), tf.float32)

        for key2, expression in value1.items():
            expression = str(expression)
            expression = add_minux(expression.replace(' ', ''))
            simplify_value2 = build_ast_all(expression)
            heads_mul[key2] *= tf.pow(simplify_value2.to_tfOp(), simplify_key1)
            # heads_mul[key2] *= tf.where(simplify_key1, simplify_value2.to_tfOp(), tf.ones_like(note_var['batch_size'], dtype=tf.float32))
    print('done3', '=================================information========================================')

    print(heads_mul)
    


    print('start', '=================================saved_model========================================')
    # stack in gpu
    outputsALL = tf.concat([[heads_base[item], heads_base[item]*heads_mul[item]] for item in heads], 0)


    with tf.device('/cpu:0'):
        for i, item in enumerate(heads):
            outputs[item+'@base'] = tf.saved_model.utils.build_tensor_info(outputsALL[2*i, :])
            outputs[item] = tf.saved_model.utils.build_tensor_info(outputsALL[2*i+1, :])
    outputs_lst = list(outputs.keys())

    
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.framework import graph_util

    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    builder = tf.saved_model.builder.SavedModelBuilder('/Users/liuyuguang/ast-cse/savedmodel')
    with tf.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'test_signature':signature})
        builder.save()
        


    # generate assets.extra file
    from tensorflow_serving.apis import model_pb2
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_log_pb2
    import tensorflow as tf
    import numpy as np

    examples = {}
    examples["requestFeatures"] = tf.make_tensor_proto([0]*req_nums, dtype=tf.float32)
    examples["noteFeatures"] = tf.make_tensor_proto([[1,2]]*note_nums, dtype=tf.float32)
    local_path = '/Users/liuyuguang/ast-cse/savedmodel/tf_serving_warmup_requests'
    with tf.python_io.TFRecordWriter(local_path) as writer:
        for i in range(1):
            request = predict_pb2.PredictRequest(
                model_spec=model_pb2.ModelSpec(
                    name="hf_vm_tf",
                    signature_name='test_signature'
                ),
                inputs=examples,
                output_filter=outputs_lst
            )
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request)
            )
            writer.write(log.SerializeToString())



    # # 拉真实数据测试
    # with open('/Users/user/code/test/deleteVM_sim/v5/test_curl_gpu.txt', 'r', encoding='utf-8') as f:
    #     content = f.read().split('\n')
    #     input = {}
        
    #     print(input)
    #     # for item0 in content:
    #     #     item = item0[82:-1]
    #     #     # print(item)
    #     #     name, value = item.split(' [')
    #     #     if name == 'LAMBDA_REQUEST_CONTEXT':
    #     #         continue
    #     #     values = list(map(float, value.split(' ')))
    #     #     input[flag_dict['raw_variable'][name]] = values
    #     outputs = []
    #     for item in heads:
    #         # outputs.append(heads_base[item])
    #         outputs.append(heads_mul[item])
    #     sess = tf.Session()
    #     print(req_nums, note_nums)
    #     values = sess.run(outputs, feed_dict={requestFeatures:[0]*req_nums, noteFeatures:[[1,2]]*note_nums})
    #     # print(values)
    #     for i, value in enumerate(values):
    #         print(list(heads)[i], " :  ", value)



