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
        match self:
            case Regex.ProperCase:
                return "p"
            case Regex.CAPS:
                return "C"
            case Regex.Lowercase:
                return "l"
            case Regex.Digits:
                return "d"
            case Regex.Alphabets:
                return "α"
            case Regex.Alphanumeric:
                return "αn"
            case Regex.Whitespace:
                return "ws"
            case Regex.StartT:
                return "∧"
            case Regex.EndT:
                return "$"
            case Regex.ProperCaseWSpaces:
                return "ps"
            case Regex.CAPSWSpaces:
                return "Cs"
            case Regex.LowercaseWSpaces:
                return "ls"
            case Regex.AlphabetsWSpaces:
                return "αs"


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

    def __repr__(self):  # pragma: no cover
        return f"Var({self.i})"


@dataclass(frozen=True)
class MatchPos(PositionExpr):
    token: Regex | str
    k: int
    dir: Dir

    def __repr__(self):  # pragma: no cover
        tok = self.token
        tok_str = repr(tok) if isinstance(tok, Regex) else f'"{tok}"'
        return f"MatchPos({tok_str}, {self.k}, {self.dir})"


@dataclass(frozen=True)
class ConstantPos(PositionExpr):
    k: int

    def __repr__(self):  # pragma: no cover
        return f"ConstantPos({self.k})"


@dataclass(frozen=True)
class ConstantStr(SubstringExpr):
    s: str

    def __repr__(self):  # pragma: no cover
        return f"ConstantStr({self.s})"


@dataclass(frozen=True)
class SubStr(SubstringExpr):
    var: Var
    lexpr: PositionExpr
    rexpr: PositionExpr

    def __repr__(self):  # pragma: no cover
        return f"SubStr({self.var}, {self.lexpr}, {self.rexpr})"


@dataclass(frozen=True)
class Concat(Expr):
    exprs: tuple[SubstringExpr, ...]

    def __repr__(self):  # pragma: no cover
        exprs = ", ".join([repr(e) for e in self.exprs])
        return f"Concat({exprs})"


# Eval

_constant_str_patterns: dict[str, re.Pattern[str]] = {}
_match_cache: dict[tuple[str, str], list[TokenMatch]] = dict()


def find_matches(string: str, token: Regex | str) -> list[TokenMatch]:
    key = (string, str(token))
    result = _match_cache.get(key)
    if result is not None:
        return result

    if isinstance(token, Regex):
        pattern = TOKEN_PATTERNS[token]
    else:
        assert isinstance(token, str)
        pattern = _constant_str_patterns.get(token)
        if pattern is None:
            pattern = re.compile(re.escape(token))
            _constant_str_patterns[token] = pattern

    matches: list[TokenMatch] = []
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


def substr(string: str, start: int, end: int) -> str:
    return string[start - 1 : end - 1]


def eval_pos(s: str, expr: PositionExpr) -> int:
    match expr:
        case ConstantPos(k):
            return k if k > 0 else len(s) + k
        case MatchPos(token, k, dir):
            matches = find_matches(s, token)
            i = k - 1 if k > 0 else len(matches) + k
            m = matches[i]
            return m.start if dir is Dir.Start else m.end
        case _:
            raise ValueError(f"Unsupported expression: {expr}")


def eval_expr(expr: Expr, env: dict[int, str]) -> str:
    match expr:
        case Concat(exprs):
            s = "".join([eval_expr(e, env) for e in exprs])
            return s
        case ConstantStr(s):
            return s
        case SubStr(Var(i), lexpr, rexpr):
            s = env[i]
            lpos = eval_pos(s, lexpr)
            rpos = eval_pos(s, rexpr)
            return substr(s, lpos, rpos)
        case _:
            raise ValueError(f"Unsupported expression: {expr}")


def eval_program(expr: Expr, input_str: str):
    env: dict[int, str] = {1: input_str}
    return eval_expr(expr, env)
