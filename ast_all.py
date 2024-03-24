# -*- coding: utf-8 -*-
# contact: log_whistle@163.com
"""abstract syntax tree with only && || ( )"""

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


class Tree(object):
    def __init__(self, value, children):
        assert (value.type in [VAR, FUNC] and len(children) == 0) or (value.type == NOT and len(children) == 1)or (value.value == LOG and len(children) == 1)\
               or (value.type == MINUS and len(children) == 1) or (value.value == SQRT and len(children) == 1) or len(children) > 1
        self.op = value
        if self.op.value in [DEV, MAX, MIN, IF, POW, WILSON, SLOGAN1, SLOGAN2, SLOGAN3, SLOGAN4, SLOGAN5, PCONT, PWL3, PWL2]:
            self.args = children
        else:
            self.args = tuple(sorted(children, key=lambda x: x.to_expression()))
        self._hash = None
        self._expression = None

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
            print('please contact log_whistle@163.com to add new function')
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
    # test single example
    condition = '(noteImpressionNoFilter && a1) && !is_video && click_limit && (A && B() || !(C && D)) && a(1) < 0 && noteIsVideo == 0'
    condition = "((register_28d && hf_click_opt_enable_fe_flag == 1) || (!register_28d && hf_click_opt_enable_fe_flag == 2)) && viewer_gender_man && user_feed_today_click_lt_1 && userDayTotalImpression > fe_day_imp_thresh && is_normal_content"

    # for math value
    condition = "0.26 * POW(MAX(fan_count_exp, 1.0)*author_ctr, -0.1262) * 0.26+1.0+-2"
    condition = '-0.26*POW(MAX(fan_count_exp,1.0),-0.1262)'
    condition = "-a*(b+c)+d+e*f"
    condition = 'IF(x*2>1&&y, -a*c/b*d+c+0+0*d+MAX(MIN(1,2),WILSON_SCORE(3, 4))*SQRT(WILSON_SCORE(3, 4))-2, value2)'
    condition = 'IF(warmup_pred_value_max_imp_1000 <= 0, 100, MAX(0.0, MIN(1.0, predValueCtr)) * warmup_pred_value_max_imp_1000)'
    condition = 'sincerityModelScore > -33.24 && sincerityModelScore <= 48.64'
    condition = 'capaPhotoAlbumId > 0 && hf_photo_album_mv_boost_hf_1000 > 0 && noteQuality > 1 && !note_hide_gt_avg_5_times'
    condition = '1.0 + MIN(1.0, 10 * PROPORTIONAL_CONTROL(feed_impression, 1000, note_age_in_secs, 172800))'
    condition = '0.9 * (1.0 - 0.9 * PIECEWISE_LINEAR_2P(nasty_note_ratio, .1, 0.3, 1.0, 0, 0.8, 1.0))'
    condition = '0.12 + MAX(0.0, 0.08 * (0.11 - user_week_rt_video_click_ratio) / 0.11)'

    condition = condition.replace(' ', '')
    print(condition)

    # for binary in [True, False]:
    for binary in [False]:
        # print('binary %s' % binary)
        condition = add_minux(condition)
        print('before: ', condition)
        ast_tree = build_ast_all(condition, binary)
        # print every node
        ast_tree.visit()
        condition = ast_tree.to_expression()
        condition = cut_minux(condition)
        print('after: ', condition)

 
