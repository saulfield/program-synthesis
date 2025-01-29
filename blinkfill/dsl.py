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

    def __repr__(self):
        # fmt: off
        match self:
            case Regex.ProperCase: return "p"
            case Regex.CAPS: return "C"
            case Regex.Lowercase: return "l"
            case Regex.Digits: return "d"
            case Regex.Alphabets: return "α"
            case Regex.Alphanumeric: return "αn"
            case Regex.Whitespace: return "ws"
            case Regex.StartT: return "∧"
            case Regex.EndT: return "$"
            case Regex.ProperCaseWSpaces: return "ps"
            case Regex.CAPSWSpaces: return "Cs"
            case Regex.LowercaseWSpaces: return "ls"
            case Regex.AlphabetsWSpaces: return "αs"
        # fmt: on


TOKEN_PATTERNS = {
    Regex.ProperCase: re.compile(r"[A-Z][a-z]+"),
    Regex.CAPS: re.compile(r"[A-Z]+"),
    Regex.Lowercase: re.compile(r"[a-z]+"),
    Regex.Digits: re.compile(r"\d+"),
    Regex.Alphabets: re.compile(r"[A-Za-z]+"),
    Regex.Alphanumeric: re.compile(r"[A-Za-z0-9]+"),
    Regex.Whitespace: re.compile(r"\s+"),
    Regex.StartT: re.compile(r"^"),
    Regex.EndT: re.compile(r"$"),
    Regex.ProperCaseWSpaces: re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"),
    Regex.CAPSWSpaces: re.compile(r"[A-Z]+(?:\s+[A-Z]+)*"),
    Regex.LowercaseWSpaces: re.compile(r"[a-z]+(?:\s+[a-z]+)*"),
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


_match_cache: dict[tuple[str, str], list[TokenMatch]] = dict()


def find_matches(string: str, token: Regex | str) -> list[TokenMatch]:
    key = (string, str(token))
    matches = _match_cache.get(key)
    if matches is not None:
        return matches

    if isinstance(token, Regex):
        pattern = TOKEN_PATTERNS[token]
    else:
        assert type(token) == str
        pattern = get_constant_str_pattern(token)

    matches = []
    for k, match in enumerate(pattern.finditer(string), 1):
        matches.append(
            TokenMatch(
                k=k,
                substr=match.group(),
                start=match.start() + 1,
                end=match.end() + 1,
            )
        )

    _match_cache[key] = matches
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
