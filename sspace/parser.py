

EOF = '\0'


class Lexer:
    """Split a string into a stream of token for easier parsing

    Examples
    --------

    >>> for tok in Lexer("arg~uniform(0, 1.0)"):
    ...     print(tok)
    (1, 'arg')
    (0, '~')
    (1, 'uniform')
    (2, '(')
    (3, '0')
    (2, ',')
    (2, ' ')
    (3, '1.0')
    (2, ')')

    """
    OPERATORS = ('~', '=' , '!', '>', '<', '&', '|')
    SEPARATORS = (',', '(', ')', EOF, ' ')

    Operator = 0
    Identifier = 1
    Separator = 2
    Number = 3

    def __init__(self, cmd):
        self.cmd = cmd
        self.pos = 0
        self.mode = Lexer.Identifier
        self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def nextc(self):
        self.pos += 1

        if self.pos >= len(self.cmd):
            return EOF

        return self.cmd[self.pos]

    def getc(self):
        if self.pos >= len(self.cmd):
            return EOF

        return self.cmd[self.pos]

    def token(self, type):
        val = ''.join(self.buffer)
        self.buffer = []
        return (type, val)

    def next(self):
        c: str = self.getc()

        if c == EOF:
            raise StopIteration()

        if c in Lexer.OPERATORS:
            while c in Lexer.OPERATORS and len(self.buffer) < 2:
                self.buffer.append(c)
                c = self.nextc()

            return self.token(Lexer.Operator)

        if c in self.SEPARATORS:
            self.buffer.append(c)
            self.nextc()
            return self.token(Lexer.Separator)

        if c.isalpha():
            self.buffer.append(c)
            c = self.nextc()

            while c.isalnum():
                self.buffer.append(c)
                c = self.nextc()

            return self.token(Lexer.Identifier)

        if c.isdigit():
            self.buffer.append(c)
            c = self.nextc()
            decimal = False

            while c.isdigit() or c == '.':
                if c == '.':
                    if decimal:
                        raise RuntimeError("Invalid number")

                    decimal = True

                self.buffer.append(c)
                c = self.nextc()

            return self.token(Lexer.Number)

        raise RuntimeError(f'Unsupported character {c}')
