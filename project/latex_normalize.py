import re


TRAILING_PUNCT_RE = re.compile(r"[.,;:]+$")
MULTISPACE_RE = re.compile(r"\s+")
SPACED_TOKENS_RE = re.compile(r"\s*([{}_^(),=+\-*/<>])\s*")


def _strip_outer_math_wrappers(s: str) -> str:
    s = s.strip()

    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()

    if s.startswith(r"\(") and s.endswith(r"\)") and len(s) >= 4:
        s = s[2:-2].strip()

    return s


def _normalize_frac_double_braces(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(
            r"\\frac\s*\{\{([^{}]+)\}\}\s*\{\{([^{}]+)\}\}",
            r"\\frac{\1}{\2}",
            s,
        )
    return s


def normalize_latex(s: str) -> str:
    s = (s or "").strip()
    s = _strip_outer_math_wrappers(s)

    s = s.replace("\r\n", " ").replace("\n", " ")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    s = s.replace(r"\\,", "")
    s = s.replace(r"\,", "")
    s = s.replace(r"\\;", "")
    s = s.replace(r"\;", "")
    s = s.replace(r"\\!", "")
    s = s.replace(r"\!", "")

    s = MULTISPACE_RE.sub(" ", s).strip()
    s = TRAILING_PUNCT_RE.sub("", s).strip()

    s = _normalize_frac_double_braces(s)

    # убираем пробелы вокруг большинства структурных токенов
    s = SPACED_TOKENS_RE.sub(r"\1", s)

    # схлопываем ещё раз на всякий случай
    s = MULTISPACE_RE.sub(" ", s).strip()

    return s


def normalized_exact_match(a: str, b: str) -> int:
    return int(normalize_latex(a) == normalize_latex(b))