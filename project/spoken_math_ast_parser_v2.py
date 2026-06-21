from __future__ import annotations

from dataclasses import dataclass
import re


class ParseError(Exception):
    pass


@dataclass
class Node:
    kind: str


@dataclass
class Symbol(Node):
    value: str

    def __init__(self, value: str):
        super().__init__("symbol")
        self.value = value


@dataclass
class Unary(Node):
    op: str
    child: Node

    def __init__(self, op: str, child: Node):
        super().__init__("unary")
        self.op = op
        self.child = child


@dataclass
class Binary(Node):
    op: str
    left: Node
    right: Node

    def __init__(self, op: str, left: Node, right: Node):
        super().__init__("binary")
        self.op = op
        self.left = left
        self.right = right


@dataclass
class Relation(Node):
    op: str
    left: Node
    right: Node

    def __init__(self, op: str, left: Node, right: Node):
        super().__init__("relation")
        self.op = op
        self.left = left
        self.right = right


@dataclass
class Fraction(Node):
    num: Node
    den: Node

    def __init__(self, num: Node, den: Node):
        super().__init__("fraction")
        self.num = num
        self.den = den


@dataclass
class Func(Node):
    name: str
    arg: Node
    base: Node | None = None

    def __init__(self, name: str, arg: Node, base: Node | None = None):
        super().__init__("func")
        self.name = name
        self.arg = arg
        self.base = base


@dataclass
class Power(Node):
    base: Node
    exp: Node

    def __init__(self, base: Node, exp: Node):
        super().__init__("power")
        self.base = base
        self.exp = exp


@dataclass
class Subscript(Node):
    base: Node
    idx: Node

    def __init__(self, base: Node, idx: Node):
        super().__init__("subscript")
        self.base = base
        self.idx = idx


@dataclass
class Derivative(Node):
    var: Node
    expr: Node
    partial: bool = False

    def __init__(self, var: Node, expr: Node, partial: bool = False):
        super().__init__("derivative")
        self.var = var
        self.expr = expr
        self.partial = partial


@dataclass
class Limit(Node):
    var: Node
    dest: Node
    expr: Node

    def __init__(self, var: Node, dest: Node, expr: Node):
        super().__init__("limit")
        self.var = var
        self.dest = dest
        self.expr = expr


@dataclass
class SumNode(Node):
    idx: Node
    start: Node
    end: Node
    expr: Node

    def __init__(self, idx: Node, start: Node, end: Node, expr: Node):
        super().__init__("sum")
        self.idx = idx
        self.start = start
        self.end = end
        self.expr = expr


@dataclass
class Call(Node):
    func: Node
    args: list[Node]

    def __init__(self, func: Node, args: list[Node]):
        super().__init__("call")
        self.func = func
        self.args = args


REL_OPS = {
    "EQ": "=",
    "GT": ">",
    "LT": "<",
    "GE": r"\geq",
    "LE": r"\leq",
    "SUBSET": r"\subset",
    "STRICT_SUBSET": r"\subset",
}

TRIG_TOKENS = {
    "SIN": r"\sin",
    "COS": r"\cos",
    "TAN": r"\tan",
    "COT": r"\cot",
    "SEC": r"\sec",
    "CSC": r"\csc",
    "ARCSIN": r"\arcsin",
    "ARCCOS": r"\arccos",
    "ARCTAN": r"\arctan",
}

WORD_NUMBERS = {
    "ноль": "0",
    "один": "1",
    "одна": "1",
    "одно": "1",
    "два": "2",
    "две": "2",
    "три": "3",
    "четыре": "4",
    "пять": "5",
    "шесть": "6",
    "семь": "7",
    "восемь": "8",
    "девять": "9",
    "десять": "10",
    "одиннадцать": "11",
    "двенадцать": "12",
    "тринадцать": "13",
    "четырнадцать": "14",
    "пятнадцать": "15",
    "шестнадцать": "16",
    "семнадцать": "17",
    "восемнадцать": "18",
    "девятнадцать": "19",
    "двадцать": "20",
    "тридцать": "30",
    "сорок": "40",
    "пятьдесят": "50",
    "шестьдесят": "60",
    "семьдесят": "70",
    "восемьдесят": "80",
    "девяносто": "90",
    "сто": "100",
}

LATIN = {
    "икс": "x",
    "игрек": "y",
    "зет": "z",
    "а": "a",
    "бэ": "b",
    "вэ": "v",
    "гэ": "g",
    "дэ": "d",
    "эф": "f",
    "же": "j",
    "жэ": "g",
    "ка": "k",
    "эль": "l",
    "эм": "m",
    "эн": "n",
    "о": "o",
    "пэ": "p",
    "ку": "q",
    "эр": "r",
    "эс": "s",
    "тэ": "t",
    "у": "u",
    "аш": "h",
    "це": "c",
    "цэ": "c",
    "дабл-ю": "w",
    "даблю": "w",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y": "y",
    "z": "z",
}

UPPER_LATIN_HINTS = {
    "а": "A",
    "бэ": "B",
    "вэ": "V",
    "гэ": "G",
    "дэ": "D",
    "эф": "F",
    "же": "J",
    "ка": "K",
    "эль": "L",
    "эм": "M",
    "эн": "N",
    "о": "O",
    "пэ": "P",
    "ку": "Q",
    "эр": "R",
    "эс": "S",
    "тэ": "T",
    "у": "U",
    "аш": "H",
    "це": "C",
    "дабл-ю": "W",
    "даблю": "W",
    "икс": "X",
    "игрек": "Y",
    "зет": "Z",
}

GREEK = {
    "альфа": r"\alpha",
    "бета": r"\beta",
    "гамма": r"\gamma",
    "дельта": r"\delta",
    "эпсилон": r"\epsilon",
    "дзета": r"\zeta",
    "эта": r"\eta",
    "тета": r"\theta",
    "йота": r"\iota",
    "каппа": r"\kappa",
    "лямбда": r"\lambda",
    "мю": r"\mu",
    "ню": r"\nu",
    "кси": r"\xi",
    "хи": r"\chi",
    "пи": r"\pi",
    "ро": r"\rho",
    "сигма": r"\sigma",
    "тау": r"\tau",
    "фи": r"\phi",
    "пси": r"\psi",
    "омега": r"\omega",
}

UPPER_GREEK = {
    "гамма": r"\Gamma",
    "дельта": r"\Delta",
    "лямбда": r"\Lambda",
    "кси": r"\Xi",
    "пи": r"\Pi",
    "сигма": r"\Sigma",
    "фи": r"\Phi",
    "пси": r"\Psi",
    "омега": r"\Omega",
    "тета": r"\Theta",
}

CONSTANTS = {
    "е": r"\mathrm\e",
    "пи": r"\pi",
}

VEC_HINTS = {"вектор", "векторный"}
BOLD_HINTS = {"жирный", "полужирный"}
MATHBB_HINTS = {"математическая", "дабл", "двойная"}
CALL_HINTS = {"при", "в", "на"}

FILLERS = {
    "это",
    "есть",
    "нам",
    "дает",
    "даёт",
    "где",
    "который",
    "которая",
    "которые",
    "что",
}


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"[.,;:!?]+", " ", text)
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


def apply_phrase_replacements(text: str) -> str:
    replacements = [
        ("частная производная по", " PDERIVBY "),
        ("производная по", " DERIVBY "),
        ("предел при", " LIMITWHEN "),
        ("стремится к", " TO "),
        ("сумма от", " SUMFROM "),
        ("натуральный логарифм", " NATLOG "),
        ("логарифм по основанию", " LOGBASEFIRST "),
        ("по основанию", " BASEOF "),
        ("квадратный корень из", " SQRT "),
        ("корень из", " SQRT "),
        ("строгое подмножество", " STRICT_SUBSET "),
        ("подмножество", " SUBSET "),
        ("больше или равно", " GE "),
        ("меньше или равно", " LE "),
        ("деленное на", " DIVBY "),
        ("деленного на", " DIVBY "),
        ("делённое на", " DIVBY "),
        ("делённого на", " DIVBY "),
        ("разделенное на", " DIVBY "),
        ("разделенного на", " DIVBY "),
        ("разделённое на", " DIVBY "),
        ("разделённого на", " DIVBY "),
        ("умножить на", " TIMES "),
        ("умноженное на", " TIMES "),
        ("в квадрате", " POW2 "),
        ("в кубе", " POW3 "),
        ("в четвертой степени", " POW4 "),
        ("в пятой степени", " POW5 "),
        ("в степени", " POW "),
        ("с индексом", " INDEX "),
        ("нижний индекс", " INDEX "),
        ("под индексом", " INDEX "),
        ("внизу", " INDEX "),
        ("открывающая скобка", " LPAR "),
        ("закрывающая скобка", " RPAR "),
        ("запятая", " COMMA "),
        ("равняется", " EQ "),
        ("это равно", " EQ "),
        ("равно", " EQ "),
        ("больше", " GT "),
        ("меньше", " LT "),
        ("до", " UNTIL "),
        ("от", " OF "),
        ("плюс", " PLUS "),
        ("минус", " MINUS "),
        ("синус", " SIN "),
        ("косинус", " COS "),
        ("тангенс", " TAN "),
        ("котангенс", " COT "),
        ("секанс", " SEC "),
        ("косеканс", " CSC "),
        ("арксинус", " ARCSIN "),
        ("арккосинус", " ARCCOS "),
        ("арктангенс", " ARCTAN "),
    ]

    text = f" {text} "
    for src, dst in replacements:
        text = text.replace(f" {src} ", dst)
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    text = apply_phrase_replacements(text)
    if not text:
        return []
    tokens = text.split()
    return [t for t in tokens if t not in FILLERS]


def symbol_from_token(tok: str, upper: bool = False) -> str | None:
    if tok in WORD_NUMBERS:
        return WORD_NUMBERS[tok]

    if re.fullmatch(r"-?\d+(\.\d+)?", tok):
        return tok

    if tok in CONSTANTS:
        return CONSTANTS[tok]

    if upper and tok in UPPER_GREEK:
        return UPPER_GREEK[tok]

    if tok in GREEK:
        return GREEK[tok]

    if upper and tok in UPPER_LATIN_HINTS:
        return UPPER_LATIN_HINTS[tok]

    if tok in LATIN:
        return LATIN[tok]

    return None


class Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def peek(self, k: int = 1) -> str | None:
        idx = self.pos + k
        if idx >= len(self.tokens):
            return None
        return self.tokens[idx]

    def advance(self) -> str:
        tok = self.current()
        if tok is None:
            raise ParseError("Unexpected end of input")
        self.pos += 1
        return tok

    def accept(self, tok: str) -> bool:
        if self.current() == tok:
            self.pos += 1
            return True
        return False

    def expect(self, tok: str) -> None:
        if not self.accept(tok):
            raise ParseError(f"Expected token {tok}, got {self.current()}")

    def is_factor_start(self, tok: str | None) -> bool:
        if tok is None:
            return False
        if tok in {
            "LPAR", "MINUS", "SQRT", "NATLOG", "LOGBASEFIRST", "DERIVBY", "PDERIVBY",
            "LIMITWHEN", "SUMFROM", "дробь"
        }:
            return True
        if tok in TRIG_TOKENS:
            return True
        if tok in VEC_HINTS or tok in BOLD_HINTS or tok in MATHBB_HINTS:
            return True
        return symbol_from_token(tok) is not None or symbol_from_token(tok, upper=True) is not None

    def parse(self) -> Node:
        node = self.parse_relation(stop=set())
        if self.current() is not None:
            raise ParseError(f"Unconsumed tokens starting from {self.current()}")
        return node

    def parse_relation(self, stop: set[str]) -> Node:
        rel_stops = stop | set(REL_OPS.keys())
        left = self.parse_add(stop=rel_stops)

        while self.current() in REL_OPS and self.current() not in stop:
            op_tok = self.advance()
            right = self.parse_add(stop=rel_stops)
            left = Relation(REL_OPS[op_tok], left, right)

        return left

    def parse_add(self, stop: set[str]) -> Node:
        add_stops = stop | {"PLUS", "MINUS"}
        left = self.parse_mul(stop=add_stops)

        while self.current() in {"PLUS", "MINUS"} and self.current() not in stop:
            op_tok = self.advance()
            right = self.parse_mul(stop=add_stops)
            left = Binary("+" if op_tok == "PLUS" else "-", left, right)

        return left

    def parse_mul(self, stop: set[str]) -> Node:
        mul_stops = stop | {"TIMES"}
        left = self.parse_div(stop=mul_stops)

        while True:
            tok = self.current()

            if tok == "TIMES" and tok not in stop:
                self.advance()
                right = self.parse_div(stop=mul_stops)
                left = Binary("*", left, right)
                continue

            if tok is not None and tok not in stop and self.is_factor_start(tok):
                right = self.parse_div(stop=mul_stops)
                left = Binary("*", left, right)
                continue

            break

        return left

    def parse_div(self, stop: set[str]) -> Node:
        div_stops = stop | {"DIVBY"}
        left = self.parse_postfix(stop=div_stops)

        while self.current() == "DIVBY" and self.current() not in stop:
            self.advance()
            right = self.parse_postfix(stop=div_stops)
            left = Fraction(left, right)

        return left

    def parse_postfix(self, stop: set[str]) -> Node:
        postfix_stops = stop | {"INDEX", "POW", "POW2", "POW3", "POW4", "POW5", "COMMA"}
        node = self.parse_prefix(stop=postfix_stops)

        while True:
            tok = self.current()

            if tok == "INDEX":
                self.advance()
                idx = self.parse_relation(stop=postfix_stops)
                node = Subscript(node, idx)
                continue

            if tok == "POW2":
                self.advance()
                node = Power(node, Symbol("2"))
                continue

            if tok == "POW3":
                self.advance()
                node = Power(node, Symbol("3"))
                continue

            if tok == "POW4":
                self.advance()
                node = Power(node, Symbol("4"))
                continue

            if tok == "POW5":
                self.advance()
                node = Power(node, Symbol("5"))
                continue

            if tok == "POW":
                self.advance()
                exp = self.parse_relation(stop=postfix_stops)
                node = Power(node, exp)
                continue

            if tok in {"OF", "при", "в", "на"}:
                save_pos = self.pos
                self.advance()

                try:
                    args = [self.parse_relation(stop=stop | {"COMMA"})]
                    while self.current() == "COMMA":
                        self.advance()
                        args.append(self.parse_relation(stop=stop | {"COMMA"}))
                    node = Call(node, args)
                    continue
                except ParseError:
                    self.pos = save_pos

            break

        return node

    def parse_prefix(self, stop: set[str]) -> Node:
        tok = self.current()

        if tok is None:
            raise ParseError("Unexpected end of input in prefix")

        if tok == "MINUS":
            self.advance()
            child = self.parse_prefix(stop=stop)
            return Unary("-", child)

        if tok == "LPAR":
            self.advance()
            node = self.parse_relation(stop=stop | {"RPAR"})
            self.expect("RPAR")
            return node

        if tok == "DERIVBY":
            self.advance()
            var = self.parse_relation(stop={"OF"})
            self.expect("OF")
            expr = self.parse_relation(stop=stop)
            return Derivative(var=var, expr=expr, partial=False)

        if tok == "PDERIVBY":
            self.advance()
            var = self.parse_relation(stop={"OF"})
            self.expect("OF")
            expr = self.parse_relation(stop=stop)
            return Derivative(var=var, expr=expr, partial=True)

        if tok == "LIMITWHEN":
            self.advance()
            var = self.parse_relation(stop={"TO"})
            self.expect("TO")
            dest = self.parse_relation(stop={"OF"})
            self.expect("OF")
            expr = self.parse_relation(stop=stop)
            return Limit(var=var, dest=dest, expr=expr)

        if tok == "SUMFROM":
            self.advance()
            idx = self.parse_relation(stop={"EQ"})
            self.expect("EQ")
            start = self.parse_relation(stop={"UNTIL"})
            self.expect("UNTIL")
            end = self.parse_relation(stop={"OF"})
            self.expect("OF")
            expr = self.parse_relation(stop=stop)
            return SumNode(idx=idx, start=start, end=end, expr=expr)

        if tok == "SQRT":
            self.advance()
            arg = self.parse_relation(stop=stop)
            return Func(name="sqrt", arg=arg)

        if tok == "NATLOG":
            self.advance()
            arg = self.parse_relation(stop=stop)
            return Func(name="ln", arg=arg)

        if tok == "LOGBASEFIRST":
            self.advance()
            base = self.parse_relation(stop={"OF"})
            self.expect("OF")
            arg = self.parse_relation(stop=stop)
            return Func(name="log", arg=arg, base=base)

        if tok in TRIG_TOKENS:
            self.advance()
            arg = self.parse_relation(stop=stop)
            return Func(name=TRIG_TOKENS[tok], arg=arg)

        if tok == "дробь":
            self.advance()
            num = self.parse_relation(stop={"DIVBY"})
            self.expect("DIVBY")
            den = self.parse_relation(stop=stop)
            return Fraction(num=num, den=den)

        if tok in VEC_HINTS:
            self.advance()
            child = self.parse_prefix(stop=stop)
            return Unary("vec", child)

        if tok in BOLD_HINTS:
            self.advance()
            child = self.parse_prefix(stop=stop)
            return Unary("mathbf", child)

        if tok in MATHBB_HINTS:
            self.advance()
            child = self.parse_prefix(stop=stop)
            return Unary("mathbb", child)

        upper_mode = False
        if tok in {"заглавная", "большая"}:
            self.advance()
            upper_mode = True
            tok = self.current()
            if tok is None:
                raise ParseError("Expected symbol after upper hint")

        sym = symbol_from_token(tok, upper=upper_mode)
        if sym is not None:
            self.advance()
            return Symbol(sym)

        raise ParseError(f"Unknown token in prefix: {tok}")


def precedence(node: Node) -> int:
    if isinstance(node, Relation):
        return 1
    if isinstance(node, Binary) and node.op in {"+", "-"}:
        return 2
    if isinstance(node, Binary) and node.op == "*":
        return 3
    if isinstance(node, Unary):
        return 4
    if isinstance(node, Derivative):
        return 5
    if isinstance(node, Limit):
        return 5
    if isinstance(node, SumNode):
        return 5
    if isinstance(node, Func):
        return 5
    if isinstance(node, Fraction):
        return 5
    if isinstance(node, Call):
        return 5
    if isinstance(node, Power):
        return 6
    if isinstance(node, Subscript):
        return 6
    return 7


def render(node: Node, parent_prec: int = 0) -> str:
    p = precedence(node)

    if isinstance(node, Symbol):
        s = node.value

    elif isinstance(node, Unary):
        child = render(node.child, p)
        if node.op == "-":
            s = f"-{child}"
        elif node.op == "vec":
            s = rf"\vec{{{render(node.child, 0)}}}"
        elif node.op == "mathbf":
            s = rf"\mathbf{{{render(node.child, 0)}}}"
        elif node.op == "mathbb":
            child_rendered = render(node.child, 0)
            s = rf"\mathbb{{{child_rendered}}}"
        else:
            raise ValueError(f"Unsupported unary op: {node.op}")

    elif isinstance(node, Binary):
        left = render(node.left, p)
        right = render(node.right, p + (1 if node.op == "*" else 0))

        if node.op == "+":
            s = f"{left}+{right}"
        elif node.op == "-":
            s = f"{left}-{right}"
        elif node.op == "*":
            s = f"{left}{right}"
        else:
            raise ValueError(f"Unsupported binary op: {node.op}")

    elif isinstance(node, Relation):
        left = render(node.left, p)
        right = render(node.right, p)
        s = f"{left}{node.op}{right}"

    elif isinstance(node, Fraction):
        num = render(node.num, 0)
        den = render(node.den, 0)
        s = rf"\frac{{{num}}}{{{den}}}"

    elif isinstance(node, Func):
        arg = render(node.arg, 0)
        if node.name == "sqrt":
            s = rf"\sqrt{{{arg}}}"
        elif node.name == "ln":
            s = rf"\ln{{{arg}}}"
        elif node.name == "log":
            base = render(node.base, 0) if node.base is not None else "?"
            s = rf"\log_{{{base}}}{{{arg}}}"
        else:
            s = rf"{node.name}{{{arg}}}"

    elif isinstance(node, Power):
        base = render(node.base, p)
        exp = render(node.exp, 0)
        s = rf"{base}^{{{exp}}}"

    elif isinstance(node, Subscript):
        base = render(node.base, p)
        idx = render(node.idx, 0)
        s = rf"{base}_{{{idx}}}"

    elif isinstance(node, Derivative):
        expr = render(node.expr, 0)
        var = render(node.var, 0)
        if node.partial:
            s = rf"\frac{{\partial}}{{\partial {var}}}{expr}"
        else:
            s = rf"\frac{{d}}{{d{var}}}{expr}"

    elif isinstance(node, Limit):
        var = render(node.var, 0)
        dest = render(node.dest, 0)
        expr = render(node.expr, 0)
        s = rf"\lim_{{{var}\to {dest}}}{expr}"

    elif isinstance(node, SumNode):
        idx = render(node.idx, 0)
        start = render(node.start, 0)
        end = render(node.end, 0)
        expr = render(node.expr, 0)
        s = rf"\sum_{{{idx}={start}}}^{{{end}}}{expr}"

    elif isinstance(node, Call):
        func = render(node.func, 0)
        args = ",".join(render(a, 0) for a in node.args)
        s = rf"{func}({args})"

    else:
        raise ValueError(f"Unsupported node type: {type(node)}")

    if p < parent_prec:
        return f"({s})"

    return s


def infer_family(node: Node) -> str:
    if isinstance(node, Fraction):
        return "fraction"
    if isinstance(node, Func):
        if node.name == "log":
            return "log_base"
        if node.name == "ln":
            return "natlog"
        if node.name == "sqrt":
            return "sqrt"
        return "trig"
    if isinstance(node, Derivative):
        return "derivative"
    if isinstance(node, Limit):
        return "limit"
    if isinstance(node, SumNode):
        return "sum"
    if isinstance(node, Power) or isinstance(node, Subscript):
        return "power_index"
    if isinstance(node, Relation):
        return "relation"
    if isinstance(node, Call):
        return "call"
    return "expr"


def tree_size(node: Node) -> int:
    if isinstance(node, Symbol):
        return 1
    if isinstance(node, Unary):
        return 1 + tree_size(node.child)
    if isinstance(node, Binary):
        return 1 + tree_size(node.left) + tree_size(node.right)
    if isinstance(node, Relation):
        return 1 + tree_size(node.left) + tree_size(node.right)
    if isinstance(node, Fraction):
        return 1 + tree_size(node.num) + tree_size(node.den)
    if isinstance(node, Func):
        return 1 + tree_size(node.arg) + (tree_size(node.base) if node.base is not None else 0)
    if isinstance(node, Power):
        return 1 + tree_size(node.base) + tree_size(node.exp)
    if isinstance(node, Subscript):
        return 1 + tree_size(node.base) + tree_size(node.idx)
    if isinstance(node, Derivative):
        return 1 + tree_size(node.var) + tree_size(node.expr)
    if isinstance(node, Limit):
        return 1 + tree_size(node.var) + tree_size(node.dest) + tree_size(node.expr)
    if isinstance(node, SumNode):
        return 1 + tree_size(node.idx) + tree_size(node.start) + tree_size(node.end) + tree_size(node.expr)
    if isinstance(node, Call):
        return 1 + tree_size(node.func) + sum(tree_size(a) for a in node.args)
    return 1


def infer_confidence(node: Node) -> float:
    family = infer_family(node)
    size = tree_size(node)

    if family in {"fraction", "log_base", "natlog", "sqrt"}:
        base = 0.99
    elif family in {"derivative", "limit", "sum"}:
        base = 0.97
    elif family == "trig":
        base = 0.96
    elif family == "call":
        base = 0.95
    elif family == "power_index":
        base = 0.93
    elif family == "relation":
        base = 0.91
    else:
        base = 0.88

    if size >= 9:
        base -= 0.03
    elif size >= 6:
        base -= 0.01

    return max(0.70, base)


def ast_parse_v2(text: str) -> tuple[str | None, str | None, float]:
    try:
        tokens = tokenize(text)
        if not tokens:
            return None, None, 0.0

        parser = Parser(tokens)
        node = parser.parse()
        latex = render(node)
        family = infer_family(node)
        conf = infer_confidence(node)
        return latex, family, conf

    except ParseError:
        return None, None, 0.0