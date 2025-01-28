import re
from abc import ABC
from enum import Enum, auto

from pydantic import PositiveInt
from pydantic.dataclasses import dataclass

# Tokens


class Regex(Enum):
    ProperCase = auto()
    CAPS = auto()
    Lowercase = auto()
    Digits = auto()
    Alphabets = auto()
    Alphanumeric = auto()
    Whitespace = auto()
    StartT = auto()
    EndT = auto()
    ProperCaseWSpaces = auto()
    CAPSWSpaces = auto()
    LowercaseWSpaces = auto()
    AlphabetsWSpaces = auto()
    ConstantStr = auto()


TOKEN_PATTERNS = {
    # ProperCase: Uppercase followed by lowercase (e.g., "Word")
    Regex.ProperCase: re.compile(r"[A-Z][a-z]+"),
    # CAPS: One or more uppercase letters
    Regex.CAPS: re.compile(r"[A-Z]+"),
    # Lowercase: One or more lowercase letters
    Regex.Lowercase: re.compile(r"[a-z]+"),
    # Digits: One or more digits
    Regex.Digits: re.compile(r"\d+"),
    # Alphabets: One or more letters (upper or lower)
    Regex.Alphabets: re.compile(r"[A-Za-z]+"),
    # Alphanumeric: One or more letters or numbers
    Regex.Alphanumeric: re.compile(r"[A-Za-z0-9]+"),
    # Whitespace: One or more whitespace characters
    Regex.Whitespace: re.compile(r"\s+"),
    # StartT: Start of string
    Regex.StartT: re.compile(r"^"),
    # EndT: End of string
    Regex.EndT: re.compile(r"$"),
    # ProperCaseWSpaces: ProperCase words with spaces between
    Regex.ProperCaseWSpaces: re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"),
    # CAPSWSpaces: CAPS words with spaces between
    Regex.CAPSWSpaces: re.compile(r"[A-Z]+(?:\s+[A-Z]+)*"),
    # LowercaseWSpaces: lowercase words with spaces between
    Regex.LowercaseWSpaces: re.compile(r"[a-z]+(?:\s+[a-z]+)*"),
    # AlphabetsWSpaces: Letter words with spaces between
    Regex.AlphabetsWSpaces: re.compile(r"[A-Za-z]+(?:\s+[A-Za-z]+)*"),
}


@dataclass
class TokenMatch:
    k: int  # Match number (1-based index)
    substr: str  # The matched substring
    start: int  # Start index
    end: int  # End index


# Grammar:
# e   = Concat(f1, · · · , fn)
# f   = ConstantStr(s)
#     | SubStr(vi, pl, pr )
# p   = (τ, k, Dir)
#     | ConstantPos(k)
# Dir = Start | End


class Dir(Enum):
    Start = 1
    End = 2


class Expr(ABC):
    pass


@dataclass(frozen=True)
class PositionExpr(Expr):
    pass


@dataclass(frozen=True)
class SubstringExpr(Expr):
    pass


@dataclass(frozen=True)
class Var(Expr):
    i: PositiveInt

    def __repr__(self):
        return f"Var({self.i})"


@dataclass(frozen=True)
class MatchPos(PositionExpr):
    token: Regex | str
    k: int
    dir: Dir


@dataclass(frozen=True)
class ConstantPos(PositionExpr):
    k: int

    def __repr__(self):
        return f"ConstantPos({self.k})"


@dataclass(frozen=True)
class ConstantStr(SubstringExpr):
    s: str

    def __repr__(self):
        return f"ConstantStr({self.s})"


@dataclass(frozen=True)
class SubStr(SubstringExpr):
    var: Var
    lexpr: PositionExpr
    rexpr: PositionExpr

    def __repr__(self):
        return f"SubStr({self.var}, {self.lexpr}, {self.rexpr})"


@dataclass(frozen=True)
class Concat(Expr):
    exprs: tuple[SubstringExpr, ...]

    def __repr__(self):
        return f"Concat{self.exprs}"


# Eval

_constant_str_patterns = {}


def get_constant_str_pattern(s: str) -> re.Pattern:
    if s not in _constant_str_patterns:
        _constant_str_patterns[s] = re.compile(re.escape(s))
    return _constant_str_patterns[s]


def find_matches(string: str, token: Regex | str) -> list[TokenMatch]:
    matches = []

    if isinstance(token, str):
        pattern = get_constant_str_pattern(token)
    else:
        assert isinstance(token, Regex)
        pattern = TOKEN_PATTERNS[token]

    for k, match in enumerate(pattern.finditer(string), 1):
        matches.append(
            TokenMatch(
                k=k,
                substr=match.group(),
                start=match.start() + 1,
                end=match.end() + 1,
            )
        )

    return matches


def token_match(string: str, token: Regex | str, k: int):
    matches = find_matches(string, token)
    i = k - 1 if k > 0 else len(matches) + k
    return matches[i]


def substr(string: str, start: int, end: int) -> str:
    return string[start - 1 : end - 1]


pos_cache: dict[PositionExpr, int] = dict()
expr_cache: dict[Expr, str] = dict()


def eval_pos(s: str, expr: PositionExpr) -> int:
    key = (s, expr)
    val = pos_cache.get(key)
    if val:
        return val
    match expr:
        case ConstantPos(k):
            val = k if k > 0 else len(s) + k
        case MatchPos(token, k, dir):
            matches = find_matches(s, token)
            i = k - 1 if k > 0 else len(matches) + k
            m = matches[i]
            val = m.start if dir is Dir.Start else m.end
        case _:
            raise ValueError(f"Unsupported expression: {expr}")
    pos_cache[key] = val
    return val


def eval_expr(env: dict[int, str], expr: Expr) -> str:
    env_hash = hash(tuple(sorted(env.items())))
    key = (env_hash, expr)
    val = expr_cache.get(key)
    if val:
        return val
    match expr:
        case Concat(exprs):
            val = "".join([eval_expr(env, e) for e in exprs])
        case ConstantStr(s):
            val = s
        case SubStr(Var(i), lexpr, rexpr):
            s = env[i]
            lpos = eval_pos(s, lexpr)
            rpos = eval_pos(s, rexpr)
            val = substr(s, lpos, rpos)
        case _:
            raise ValueError(f"Unsupported expression: {expr}")
    expr_cache[key] = val
    return val
