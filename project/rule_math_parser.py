from __future__ import annotations

import re


RELATIONS = [
    (" больше или равно ", r"\geq"),
    (" меньше или равно ", r"\leq"),
    (" строгое подмножество ", r"\subset"),
    (" подмножество ", r"\subset"),
    (" больше ", ">"),
    (" меньше ", "<"),
    (" равно ", "="),
    (" равняется ", "="),
    (" это равно ", "="),
]

NUMBERS = {
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
    "половина": r"\frac{1}{2}",
}

LATIN_VARS = {
    "икс": "x",
    "x": "x",
    "игрек": "y",
    "у": "u",
    "ю": "u",
    "y": "y",
    "зет": "z",
    "z": "z",
    "а": "a",
    "бэ": "b",
    "вэ": "v",
    "гэ": "g",
    "дэ": "d",
    "же": "j",
    "жэ": "g",
    "эф": "f",
    "эль": "l",
    "эм": "m",
    "эн": "n",
    "пэ": "p",
    "ку": "q",
    "эр": "r",
    "эс": "s",
    "тэ": "t",
    "аш": "h",
    "це": "c",
    "цэ": "c",
    "ка": "k",
    "о": "o",
    "w": "w",
    "дабл-ю": "w",
    "даблю": "w",
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

CONSTANTS = {
    "е": r"\mathrm\e",
    "пи": r"\pi",
}

FUNCTIONS = {
    "синус": r"\sin",
    "косинус": r"\cos",
    "тангенс": r"\tan",
    "котангенс": r"\cot",
    "секанс": r"\sec",
    "косеканс": r"\csc",
    "арксинус": r"\arcsin",
    "арккосинус": r"\arccos",
    "арктангенс": r"\arctan",
}

FILLER_WORDS = {
    "это",
    "есть",
    "нам",
    "дает",
    "даёт",
    "при",
    "условии",
}


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"[ \t\r\n]+", " ", text)
    text = text.strip(" .,:;!?")
    return text


def cleanup_phrase(text: str) -> str:
    text = normalize_text(text)
    parts = [p for p in text.split() if p not in FILLER_WORDS]
    return " ".join(parts).strip()


def strip_outer_brackets(text: str) -> str:
    text = text.strip()
    if text.startswith("открывающая скобка ") and text.endswith(" закрывающая скобка"):
        return text[len("открывающая скобка "):-len(" закрывающая скобка")].strip()
    return text


def parse_number(text: str) -> str | None:
    text = cleanup_phrase(text)

    if text in NUMBERS:
        return NUMBERS[text]

    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return text

    return None


def parse_symbol(text: str, prefer_upper: bool = False) -> str | None:
    text = cleanup_phrase(text)

    if text in CONSTANTS:
        return CONSTANTS[text]

    if text in GREEK:
        return GREEK[text]

    if text in LATIN_VARS:
        sym = LATIN_VARS[text]
        if prefer_upper and len(sym) == 1 and sym.isalpha():
            return sym.upper()
        return sym

    num = parse_number(text)
    if num is not None:
        return num

    return None


def maybe_parenthesize(expr: str) -> str:
    if any(op in expr for op in ["+", "-", "=", "<", ">", r"\geq", r"\leq"]):
        return f"({expr})"
    return expr


def parse_atom(text: str, prefer_upper: bool = False) -> str | None:
    text = cleanup_phrase(strip_outer_brackets(text))

    if not text:
        return None

    # vector
    m = re.fullmatch(r"вектор (.+)", text)
    if m:
        base = parse_atom(m.group(1), prefer_upper=prefer_upper)
        if base:
            return rf"\vec{{{base}}}"

    # index
    m = re.fullmatch(r"(.+?) (?:с индексом|нижний индекс|внизу) (.+)", text)
    if m:
        base = parse_atom(m.group(1), prefer_upper=prefer_upper)
        idx = parse_simple_expr(m.group(2))
        if base and idx:
            return rf"{base}_{{{idx}}}"

    # common powers
    m = re.fullmatch(r"(.+?) в квадрате", text)
    if m:
        base = parse_atom(m.group(1), prefer_upper=prefer_upper)
        if base:
            return rf"{base}^{{2}}"

    m = re.fullmatch(r"(.+?) в кубе", text)
    if m:
        base = parse_atom(m.group(1), prefer_upper=prefer_upper)
        if base:
            return rf"{base}^{{3}}"

    m = re.fullmatch(r"(.+?) в четвертой степени", text)
    if m:
        base = parse_atom(m.group(1), prefer_upper=prefer_upper)
        if base:
            return rf"{base}^{{4}}"

    m = re.fullmatch(r"(.+?) в пятой степени", text)
    if m:
        base = parse_atom(m.group(1), prefer_upper=prefer_upper)
        if base:
            return rf"{base}^{{5}}"

    m = re.fullmatch(r"(.+?) в степени (.+)", text)
    if m:
        base = parse_atom(m.group(1), prefer_upper=prefer_upper)
        exp = parse_simple_expr(m.group(2))
        if base and exp:
            return rf"{base}^{{{exp}}}"

    sym = parse_symbol(text, prefer_upper=prefer_upper)
    if sym is not None:
        return sym

    return None


def split_once(text: str, token: str):
    if token not in text:
        return None
    left, right = text.split(token, 1)
    return left.strip(), right.strip()


def parse_function(text: str) -> tuple[str | None, str | None, float]:
    text = cleanup_phrase(text)

    # derivative
    m = re.fullmatch(r"производная по (.+?) от (.+)", text)
    if m:
        var = parse_atom(m.group(1))
        expr = parse_simple_expr(m.group(2))
        if var and expr:
            return rf"\frac{{d}}{{d{var}}}{expr}", "derivative", 0.98

    m = re.fullmatch(r"частная производная по (.+?) от (.+)", text)
    if m:
        var = parse_atom(m.group(1))
        expr = parse_simple_expr(m.group(2))
        if var and expr:
            return rf"\frac{{\partial}}{{\partial {var}}}{expr}", "partial_derivative", 0.98

    # logarithms
    m = re.fullmatch(r"логарифм (.+?) по основанию (.+)", text)
    if m:
        arg = parse_simple_expr(m.group(1))
        base = parse_simple_expr(m.group(2))
        if arg and base:
            return rf"\log_{{{base}}}{{{arg}}}", "log_base", 0.98

    m = re.fullmatch(r"логарифм по основанию (.+?) от (.+)", text)
    if m:
        base = parse_simple_expr(m.group(1))
        arg = parse_simple_expr(m.group(2))
        if arg and base:
            return rf"\log_{{{base}}}{{{arg}}}", "log_base", 0.98

    m = re.fullmatch(r"натуральный логарифм (.+)", text)
    if m:
        arg = parse_simple_expr(m.group(1))
        if arg:
            return rf"\ln{{{arg}}}", "natlog", 0.98

    # roots
    m = re.fullmatch(r"(?:квадратный )?корень из (.+)", text)
    if m:
        arg = parse_simple_expr(m.group(1))
        if arg:
            return rf"\sqrt{{{arg}}}", "sqrt", 0.98

    # trig
    for ru_name, tex_name in FUNCTIONS.items():
        m = re.fullmatch(rf"{ru_name} (.+)", text)
        if m:
            arg = parse_simple_expr(m.group(1))
            if arg:
                return rf"{tex_name}{{{arg}}}", "function", 0.95

    # limit
    m = re.fullmatch(r"предел при (.+?) стремится к (.+?) от (.+)", text)
    if m:
        var = parse_atom(m.group(1))
        lim = parse_simple_expr(m.group(2))
        expr = parse_simple_expr(m.group(3))
        if var and lim and expr:
            return rf"\lim_{{{var}\to {lim}}}{expr}", "limit", 0.96

    # sum
    m = re.fullmatch(r"сумма от (.+?) равно (.+?) до (.+?) от (.+)", text)
    if m:
        idx = parse_atom(m.group(1))
        start = parse_simple_expr(m.group(2))
        end = parse_simple_expr(m.group(3))
        expr = parse_simple_expr(m.group(4))
        if idx and start and end and expr:
            return rf"\sum_{{{idx}={start}}}^{{{end}}}{expr}", "sum", 0.96

    return None, None, 0.0


def parse_fraction(text: str) -> tuple[str | None, str | None, float]:
    text = cleanup_phrase(text)

    m = re.fullmatch(r"дробь (.+?) деленн(?:ого|ое|ая|ый)? на (.+)", text)
    if m:
        num = parse_simple_expr(m.group(1))
        den = parse_simple_expr(m.group(2))
        if num and den:
            return rf"\frac{{{num}}}{{{den}}}", "fraction", 0.97

    # conservative non-"дробь" variant
    m = re.fullmatch(r"(.+?) деленн(?:ого|ое|ая|ый)? на (.+)", text)
    if m:
        left = parse_atom(m.group(1))
        right = parse_atom(m.group(2))
        if left and right:
            return rf"\frac{{{left}}}{{{right}}}", "fraction_simple", 0.91

    return None, None, 0.0


def parse_product(text: str) -> str | None:
    text = cleanup_phrase(text)

    parts = re.split(r"\s+умнож(?:енное|ить)?\s+на\s+", text)
    if len(parts) >= 2:
        rendered = [parse_atom(p) or parse_simple_expr(p) for p in parts]
        if all(rendered):
            return "".join(rendered)

    return None


def parse_add_sub(text: str) -> str | None:
    text = cleanup_phrase(text)

    if " плюс " not in text and " минус " not in text:
        return None

    parts = re.split(r"\s+(плюс|минус)\s+", text)
    if len(parts) < 3:
        return None

    out = []
    first = parse_term(parts[0])
    if not first:
        return None
    out.append(first)

    i = 1
    while i < len(parts):
        op = parts[i]
        rhs = parse_term(parts[i + 1])
        if not rhs:
            return None
        out.append("+" if op == "плюс" else "-")
        out.append(rhs)
        i += 2

    return "".join(out)


def parse_term(text: str) -> str | None:
    text = cleanup_phrase(text)

    frac, _, conf = parse_fraction(text)
    if frac and conf >= 0.91:
        return frac

    prod = parse_product(text)
    if prod:
        return prod

    atom = parse_atom(text)
    if atom:
        return atom

    func, _, conf = parse_function(text)
    if func and conf >= 0.95:
        return func

    return None


def parse_simple_expr(text: str) -> str | None:
    text = cleanup_phrase(strip_outer_brackets(text))

    if not text:
        return None

    # relation
    for token, tex_rel in RELATIONS:
        split = split_once(text, token)
        if split is not None:
            left_text, right_text = split
            left = parse_simple_expr(left_text)
            right = parse_simple_expr(right_text)

            if left and right:
                # for strict subset we prefer uppercase latin if possible
                if tex_rel == r"\subset":
                    left_u = parse_atom(left_text, prefer_upper=True)
                    right_u = parse_atom(right_text, prefer_upper=True)
                    if left_u and right_u:
                        return rf"{left_u}{tex_rel}{right_u}"
                return rf"{left}{tex_rel}{right}"

    # high-confidence function patterns
    func, _, conf = parse_function(text)
    if func and conf >= 0.95:
        return func

    # addition/subtraction
    expr = parse_add_sub(text)
    if expr:
        return expr

    # term
    term = parse_term(text)
    if term:
        return term

    return None


def rule_parse(text: str) -> tuple[str | None, str | None, float]:
    text_norm = cleanup_phrase(text)

    # direct high-confidence patterns
    expr = parse_simple_expr(text_norm)
    if expr is not None:
        # confidence heuristic
        if any(k in text_norm for k in [
            "производная", "логарифм", "корень", "сумма", "предел",
            "деленное на", "делённое на", "дробь"
        ]):
            return expr, "strict_rule", 0.97

        if "строгое подмножество" in text_norm or "подмножество" in text_norm:
            return expr, "strict_rule", 0.95

        if any(k in text_norm for k in ["в квадрате", "в кубе", "в степени", "с индексом", "нижний индекс", "внизу"]):
            return expr, "semi_strict_rule", 0.90

        if "плюс" in text_norm or "минус" in text_norm or "равно" in text_norm:
            return expr, "semi_strict_rule", 0.88

        return expr, "simple_rule", 0.80

    return None, None, 0.0