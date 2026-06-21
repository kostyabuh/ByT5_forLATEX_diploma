from __future__ import annotations

import re


MULTISPACE_RE = re.compile(r"\s+")
TRAILING_PUNCT_RE = re.compile(r"[ \t\r\n]*[.,;:]+[ \t\r\n]*$")
DOUBLE_BRACE_FRAC_RE = re.compile(
    r"\\frac\s*\{\{([^{}]+)\}\}\s*\{\{([^{}]+)\}\}"
)

COMMON_FUNCS = [
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "ln",
    "log",
    "exp",
    "sqrt",
    "lim",
    "max",
    "min",
]

COMMON_OPERATORS = [
    "deg",
    "ker",
    "rank",
    "dim",
    "supp",
    "Tr",
    "det",
    "Re",
    "Im",
]


def _strip_outer_math_wrappers(s: str) -> str:
    s = s.strip()

    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()

    if s.startswith(r"\(") and s.endswith(r"\)") and len(s) >= 4:
        s = s[2:-2].strip()

    if s.startswith(r"\[") and s.endswith(r"\]") and len(s) >= 4:
        s = s[2:-2].strip()

    return s


def _normalize_spacing_commands(s: str) -> str:
    for token in [r"\,", r"\;", r"\!", r"\quad", r"\qquad"]:
        s = s.replace(token, "")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    return s


def _normalize_frac_double_braces(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = DOUBLE_BRACE_FRAC_RE.sub(r"\\frac{\1}{\2}", s)
    return s


def _add_backslash_to_common_funcs(s: str) -> str:
    for name in COMMON_FUNCS:
        s = re.sub(rf"(?<!\\)\b{name}\b", rf"\\{name}", s)

    # для операторов делаем operatorname, если вдруг модель выдала bare token
    for name in COMMON_OPERATORS:
        s = re.sub(
            rf"(?<![\\A-Za-z]){name}(?![A-Za-z])",
            rf"\\operatorname{{{name}}}",
            s,
        )

    return s


def _normalize_differentials(s: str) -> str:
    s = re.sub(r"\\mathrm\s*e", r"\\mathrm\\e", s)
    s = re.sub(r"\\mathrm\{e\}", r"\\mathrm\\e", s)
    return s


def _fix_brace_whitespace(s: str) -> str:
    s = re.sub(r"\{\s+", "{", s)
    s = re.sub(r"\s+\}", "}", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    return s


def _collapse_spaces(s: str) -> str:
    return MULTISPACE_RE.sub(" ", s).strip()


def _trim_trailing_punct(s: str) -> str:
    return TRAILING_PUNCT_RE.sub("", s).strip()


def postprocess_latex_prediction(s: str) -> str:
    s = (s or "").strip()
    s = _strip_outer_math_wrappers(s)
    s = s.replace("\r\n", " ").replace("\n", " ")
    s = _normalize_spacing_commands(s)
    s = _normalize_frac_double_braces(s)
    s = _add_backslash_to_common_funcs(s)
    s = _normalize_differentials(s)
    s = _fix_brace_whitespace(s)
    s = _collapse_spaces(s)
    s = _trim_trailing_punct(s)
    s = _collapse_spaces(s)
    return s


if __name__ == "__main__":
    samples = [
        r"\frac{d}{dx}csc(x)",
        r"\log_{a}(x).",
        r" \operatorname{supp}T \subset \bigcup_{i=1}^{s} L_{P_i}. ",
    ]
    for x in samples:
        print("IN :", x)
        print("OUT:", postprocess_latex_prediction(x))
        print()