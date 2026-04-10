from __future__ import annotations

import ast
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
except Exception as exc:
    cp = None
    _CUPY_IMPORT_ERROR = exc
    _CUDA_BACKEND_NAME = None
    _HAVE_CUDA = False
    _CUDA_STATUS = f"Unavailable ({exc.__class__.__name__}: {exc})"
else:
    try:
        probe = cp.asarray(np.arange(4, dtype=np.float64).reshape(2, 2))
        cp.asnumpy(cp.transpose(probe).reshape(4))
    except Exception as exc:
        cp = None
        _CUPY_IMPORT_ERROR = exc
        _CUDA_BACKEND_NAME = None
        _HAVE_CUDA = False
        _CUDA_STATUS = f"Unavailable (CuPy runtime check failed: {exc.__class__.__name__}: {exc})"
    else:
        _CUPY_IMPORT_ERROR = None
        _CUDA_BACKEND_NAME = "cupy"
        _HAVE_CUDA = True
        _CUDA_STATUS = "Available (CuPy)"


Array = np.ndarray
_EPS = 1e-12
_BUILTIN_INCLUDES = {"qelib1.inc", "stdgates.inc"}
_MAX_WHILE_ITERATIONS = 10000


def _validate_qubits(n_qubits: int, qs: Sequence[int]):
    for q in qs:
        if not (0 <= q < n_qubits):
            raise IndexError(f"Qubit index {q} out of range for {n_qubits} qubits")


def _as_tuple(x: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)


def _split_csv(text: str) -> List[str]:
    return _split_top_level(text, split_on_whitespace=False)


def _split_operands(text: str) -> List[str]:
    return _split_top_level(text, split_on_whitespace=True)


def _split_top_level(text: str, split_on_whitespace: bool) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_string = False
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            current.append(ch)
            if ch == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                in_string = False
            i += 1
            continue

        if ch in ("'", '"'):
            in_string = True
            quote = ch
            current.append(ch)
            i += 1
            continue

        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1

        at_top = depth_paren == 0 and depth_bracket == 0 and depth_brace == 0
        if at_top and ch == ",":
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            i += 1
            continue

        if at_top and split_on_whitespace and ch.isspace():
            part = "".join(current).strip()
            if part:
                parts.append(part)
                current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _strip_comments(text: str) -> str:
    out: List[str] = []
    i = 0
    in_string = False
    quote = ""
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                in_string = False
            i += 1
            continue

        if ch in ("'", '"'):
            in_string = True
            quote = ch
            out.append(ch)
            i += 1
            continue

        if text.startswith("//", i):
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "#":
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _tokenize_program(text: str) -> List[str]:
    text = _strip_comments(text)
    tokens: List[str] = []
    current: List[str] = []
    in_string = False
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            current.append(ch)
            if ch == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                in_string = False
            i += 1
            continue

        if ch in ("'", '"'):
            in_string = True
            quote = ch
            current.append(ch)
            i += 1
            continue

        if ch in "{};\r\n":
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            if ch in "{}":
                tokens.append(ch)
            i += 1
            continue

        current.append(ch)
        i += 1

    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _consume_balanced(text: str, open_char: str, close_char: str) -> Tuple[str, str]:
    if not text.startswith(open_char):
        raise SyntaxError(f"Expected '{open_char}' in '{text}'")
    depth = 0
    in_string = False
    quote = ""
    for idx, ch in enumerate(text):
        if in_string:
            if ch == "\\":
                continue
            if ch == quote:
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[1:idx], text[idx + 1 :]
    raise SyntaxError(f"Unbalanced '{open_char}{close_char}' in '{text}'")


def _parse_complex_qasm(tok: str) -> complex:
    t = tok.strip()
    if t.endswith("i") and "j" not in t:
        t = t[:-1] + "j"
    t = t.strip("()")
    return complex(t)


def _parse_q_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x != ""]


def _to_float(value: object) -> float:
    if isinstance(value, RegisterProxy):
        value = int(value)
    if isinstance(value, complex):
        if abs(value.imag) > _EPS:
            raise ValueError(f"Expected a real value, got {value}")
        value = value.real
    return float(value)


def _to_int(value: object) -> int:
    if isinstance(value, RegisterProxy):
        value = int(value)
    if isinstance(value, complex):
        if abs(value.imag) > _EPS:
            raise ValueError(f"Expected an integer-compatible value, got {value}")
        value = value.real
    return int(round(float(value)))


class RegisterProxy:
    def __init__(self, bits: Sequence[int]):
        self.bits = tuple(int(bit) for bit in bits)

    def __getitem__(self, item: int) -> int:
        return self.bits[int(item)]

    def __int__(self) -> int:
        value = 0
        for idx, bit in enumerate(self.bits):
            value |= (int(bit) & 1) << idx
        return value

    def __bool__(self) -> bool:
        return bool(int(self))

    def _coerce_other(self, other: object) -> int:
        if isinstance(other, RegisterProxy):
            return int(other)
        return _to_int(other)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        return int(self) == self._coerce_other(other)

    def __lt__(self, other: object) -> bool:
        return int(self) < self._coerce_other(other)

    def __le__(self, other: object) -> bool:
        return int(self) <= self._coerce_other(other)

    def __gt__(self, other: object) -> bool:
        return int(self) > self._coerce_other(other)

    def __ge__(self, other: object) -> bool:
        return int(self) >= self._coerce_other(other)

    def __repr__(self) -> str:
        return f"RegisterProxy(bits={list(self.bits)}, value={int(self)})"


_SAFE_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": abs,
    "ln": np.log,
    "log": np.log,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
}

_SAFE_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_SAFE_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _normalize_expr(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("^", "**")
    expr = expr.replace("&&", " and ")
    expr = expr.replace("||", " or ")
    expr = re.sub(r"(?<![=!<>])!(?!=)", " not ", expr)
    return expr


def _eval_expression(expr: str, symbols: Dict[str, object]) -> object:
    expr = _normalize_expr(expr)
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body, symbols)


def _eval_ast(node: ast.AST, symbols: Dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return symbols[node.id]
        if node.id == "pi":
            return math.pi
        if node.id == "tau":
            return math.tau
        if node.id == "e":
            return math.e
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        raise NameError(f"Unknown identifier '{node.id}'")
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(elt, symbols) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_eval_ast(elt, symbols) for elt in node.elts]
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        return op(_eval_ast(node.operand, symbols))
    if isinstance(node, ast.BinOp):
        op = _SAFE_BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
        return op(_eval_ast(node.left, symbols), _eval_ast(node.right, symbols))
    if isinstance(node, ast.BoolOp):
        values = [_eval_ast(v, symbols) for v in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for value in values:
                result = result and bool(value)
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in values:
                result = result or bool(value)
            return result
        raise ValueError(f"Unsupported boolean operator: {ast.dump(node.op)}")
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, symbols)
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _eval_ast(comparator, symbols)
            op = _SAFE_COMPARE_OPS.get(type(op_node))
            if op is None:
                raise ValueError(f"Unsupported comparison operator: {ast.dump(op_node)}")
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")
        func = _SAFE_FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unsupported function '{node.func.id}'")
        args = [_eval_ast(arg, symbols) for arg in node.args]
        return func(*args)
    if isinstance(node, ast.Subscript):
        value = _eval_ast(node.value, symbols)
        index = _eval_ast(node.slice, symbols)
        return value[_to_int(index)]
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@dataclass
class Statement:
    kind: str
    text: str = ""
    condition: str = ""
    var_name: str = ""
    iterable: str = ""
    name: str = ""
    params: Tuple[str, ...] = ()
    qargs: Tuple[str, ...] = ()
    body: List["Statement"] = field(default_factory=list)
    else_body: List["Statement"] = field(default_factory=list)


@dataclass
class GateDefinition:
    name: str
    params: Tuple[str, ...]
    qargs: Tuple[str, ...]
    body: List[Statement]


@dataclass
class ProgramContext:
    sim: "QuantumSimulator"
    base_path: Optional[Path] = None
    qregs: Dict[str, List[int]] = field(default_factory=dict)
    cregs: Dict[str, List[int]] = field(default_factory=dict)
    qaliases: Dict[str, List[int]] = field(default_factory=dict)
    gate_defs: Dict[str, GateDefinition] = field(default_factory=dict)
    vars: Dict[str, object] = field(default_factory=dict)
    explicit_qregs: bool = False
    explicit_cregs: bool = False
    saw_measurement: bool = False
    qasm_version: Optional[str] = None

    def child(self) -> "ProgramContext":
        return ProgramContext(
            sim=self.sim,
            base_path=self.base_path,
            qregs=self.qregs,
            cregs=self.cregs,
            qaliases=dict(self.qaliases),
            gate_defs=self.gate_defs,
            vars=dict(self.vars),
            explicit_qregs=self.explicit_qregs,
            explicit_cregs=self.explicit_cregs,
            saw_measurement=self.saw_measurement,
            qasm_version=self.qasm_version,
        )


def _parse_program(
    text: str,
    base_path: Optional[Path] = None,
    include_stack: Optional[Tuple[Path, ...]] = None,
) -> List[Statement]:
    text = text.lstrip("\ufeff")
    tokens = _tokenize_program(text)
    statements, pos = _parse_block(tokens, 0, base_path, include_stack or ())
    if pos != len(tokens):
        raise SyntaxError("Unexpected trailing tokens")
    return statements


def _parse_block(
    tokens: List[str],
    pos: int,
    base_path: Optional[Path],
    include_stack: Tuple[Path, ...],
) -> Tuple[List[Statement], int]:
    statements: List[Statement] = []
    while pos < len(tokens):
        token = tokens[pos]
        if token == "}":
            return statements, pos + 1
        parsed, pos = _parse_statement(tokens, pos, base_path, include_stack)
        statements.extend(parsed)
    return statements, pos


def _parse_statement(
    tokens: List[str],
    pos: int,
    base_path: Optional[Path],
    include_stack: Tuple[Path, ...],
) -> Tuple[List[Statement], int]:
    token = tokens[pos]
    lower = token.lower()

    if token in ("{", "}"):
        raise SyntaxError(f"Unexpected block token '{token}'")

    if lower.startswith("include "):
        path = _parse_include_path(token)
        include_path = Path(path)
        if include_path.name in _BUILTIN_INCLUDES:
            return [], pos + 1
        if include_path.is_absolute():
            raise ValueError("Absolute include paths are not supported")
        root = base_path or Path.cwd()
        resolved = (root / include_path).resolve()
        if resolved in include_stack:
            raise ValueError(f"Recursive include detected: {resolved}")
        source = resolved.read_text(encoding="utf-8")
        included = _parse_program(source, resolved.parent, include_stack + (resolved,))
        return included, pos + 1

    if lower.startswith("gate "):
        name, params, qargs = _parse_gate_header(token)
        pos += 1
        if pos >= len(tokens) or tokens[pos] != "{":
            raise SyntaxError(f"Gate '{name}' requires a braced body")
        body, pos = _parse_block(tokens, pos + 1, base_path, include_stack)
        return [Statement(kind="gate", name=name, params=params, qargs=qargs, body=body)], pos

    if lower.startswith("if"):
        stmt, pos = _parse_if_statement(tokens, pos, base_path, include_stack)
        return [stmt], pos

    if lower.startswith("for "):
        stmt, pos = _parse_for_statement(tokens, pos, base_path, include_stack)
        return [stmt], pos

    if lower.startswith("while"):
        stmt, pos = _parse_while_statement(tokens, pos, base_path, include_stack)
        return [stmt], pos

    return [Statement(kind="cmd", text=token)], pos + 1


def _parse_inline_program(text: str, base_path: Optional[Path], include_stack: Tuple[Path, ...]) -> List[Statement]:
    return _parse_program(text + ";", base_path, include_stack)


def _parse_condition_header(keyword: str, token: str) -> Tuple[str, str]:
    rest = token[len(keyword) :].lstrip()
    condition, tail = _consume_balanced(rest, "(", ")")
    return condition.strip(), tail.strip()


def _parse_if_statement(
    tokens: List[str],
    pos: int,
    base_path: Optional[Path],
    include_stack: Tuple[Path, ...],
) -> Tuple[Statement, int]:
    token = tokens[pos]
    condition, inline_tail = _parse_condition_header("if", token)
    pos += 1

    if inline_tail:
        body = _parse_inline_program(inline_tail, base_path, include_stack)
    elif pos < len(tokens) and tokens[pos] == "{":
        body, pos = _parse_block(tokens, pos + 1, base_path, include_stack)
    else:
        body, pos = _parse_statement(tokens, pos, base_path, include_stack)

    else_body: List[Statement] = []
    if pos < len(tokens) and tokens[pos].lower().startswith("else"):
        else_token = tokens[pos]
        pos += 1
        inline_else = else_token[4:].strip()
        if inline_else:
            else_body = _parse_inline_program(inline_else, base_path, include_stack)
        elif pos < len(tokens) and tokens[pos] == "{":
            else_body, pos = _parse_block(tokens, pos + 1, base_path, include_stack)
        else:
            else_body, pos = _parse_statement(tokens, pos, base_path, include_stack)

    return Statement(kind="if", condition=condition, body=body, else_body=else_body), pos


def _parse_for_statement(
    tokens: List[str],
    pos: int,
    base_path: Optional[Path],
    include_stack: Tuple[Path, ...],
) -> Tuple[Statement, int]:
    token = tokens[pos]
    rest = token[3:].strip()
    match = re.match(r"(?:(?:const\s+)?(?:int|uint(?:\[\d+\])?|integer)\s+)?([A-Za-z_]\w*)\s+in\s+(.+)$", rest)
    if match is None:
        raise SyntaxError(f"Invalid for-loop syntax: '{token}'")
    var_name = match.group(1)
    iterable_and_tail = match.group(2).strip()
    iterable, inline_tail = _split_iterable_and_tail(iterable_and_tail)
    pos += 1
    if inline_tail:
        body = _parse_inline_program(inline_tail, base_path, include_stack)
    elif pos < len(tokens) and tokens[pos] == "{":
        body, pos = _parse_block(tokens, pos + 1, base_path, include_stack)
    else:
        body, pos = _parse_statement(tokens, pos, base_path, include_stack)
    return Statement(kind="for", var_name=var_name, iterable=iterable, body=body), pos


def _parse_while_statement(
    tokens: List[str],
    pos: int,
    base_path: Optional[Path],
    include_stack: Tuple[Path, ...],
) -> Tuple[Statement, int]:
    token = tokens[pos]
    condition, inline_tail = _parse_condition_header("while", token)
    pos += 1
    if inline_tail:
        body = _parse_inline_program(inline_tail, base_path, include_stack)
    elif pos < len(tokens) and tokens[pos] == "{":
        body, pos = _parse_block(tokens, pos + 1, base_path, include_stack)
    else:
        body, pos = _parse_statement(tokens, pos, base_path, include_stack)
    return Statement(kind="while", condition=condition, body=body), pos


def _split_iterable_and_tail(text: str) -> Tuple[str, str]:
    if text.startswith("["):
        inner, tail = _consume_balanced(text, "[", "]")
        return f"[{inner}]", tail.strip()
    if text.startswith("{"):
        inner, tail = _consume_balanced(text, "{", "}")
        return f"{{{inner}}}", tail.strip()
    if text.startswith("range("):
        inner, tail = _consume_balanced(text[5:], "(", ")")
        return f"range({inner})", tail.strip()
    return text, ""


def _parse_gate_header(token: str) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    match = re.match(r"gate\s+([A-Za-z_]\w*)(?:\s*\((.*?)\))?\s+(.*)$", token)
    if match is None:
        raise SyntaxError(f"Invalid gate declaration: '{token}'")
    name = match.group(1)
    params = tuple(part.strip() for part in _split_csv(match.group(2) or "") if part.strip())
    qargs = tuple(part.strip() for part in _split_csv(match.group(3)) if part.strip())
    if not qargs:
        raise SyntaxError(f"Gate '{name}' must declare at least one qubit argument")
    return name, params, qargs


def _parse_include_path(token: str) -> str:
    match = re.match(r"include\s+(['\"])(.+?)\1$", token, flags=re.IGNORECASE)
    if match is None:
        raise SyntaxError(f"Invalid include syntax: '{token}'")
    return match.group(2)


class QuantumSimulator:
    def __init__(self, num_qubits: int, seed: Optional[int] = None, prefer_gpu: bool = True):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.prefer_gpu = bool(prefer_gpu)
        self._seed = None if seed is None else int(seed)
        self.num_qubits = int(num_qubits)
        self.dim = 1 << self.num_qubits
        self.dtype = np.complex128
        self.rng = np.random.default_rng(self._seed)
        self.state: Array = np.zeros(self.dim, dtype=self.dtype)
        self.state[0] = 1.0 + 0.0j
        self.creg: List[int] = [0] * self.num_qubits

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def backend_name(self) -> str:
        return _CUDA_BACKEND_NAME if (self.prefer_gpu and _HAVE_CUDA) else "cpu"

    @property
    def backend_status(self) -> str:
        if self.prefer_gpu and _HAVE_CUDA:
            return f"Using {_CUDA_BACKEND_NAME}"
        return "CPU only" if _HAVE_CUDA else _CUDA_STATUS

    def reseed(self, seed: Optional[int]):
        self._seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)

    def reset_simulation(self, num_qubits: int, preserve_seed: bool = True):
        seed = self._seed if preserve_seed else None
        self.__init__(num_qubits, seed=seed, prefer_gpu=self.prefer_gpu)

    def copy_from(self, other: "QuantumSimulator"):
        self.prefer_gpu = other.prefer_gpu
        self._seed = other._seed
        self.num_qubits = other.num_qubits
        self.dim = other.dim
        self.dtype = other.dtype
        self.rng = other.rng
        self.state = other.state.copy()
        self.creg = list(other.creg)

    def _axis_of(self, q: int) -> int:
        return self.num_qubits - 1 - int(q)

    def _as_tensor(self) -> Array:
        return self.state.reshape((2,) * self.num_qubits)

    def _normalize_(self):
        norm2 = float(np.vdot(self.state, self.state).real)
        if abs(norm2 - 1.0) > 1e-15 and norm2 > _EPS:
            self.state /= math.sqrt(norm2)
        self.state.real[np.abs(self.state.real) < _EPS] = 0.0
        self.state.imag[np.abs(self.state.imag) < _EPS] = 0.0

    def _gpu_enabled(self) -> bool:
        return self.prefer_gpu and _HAVE_CUDA

    def _apply_unitary_gpu(self, targets: Tuple[int, ...], U: Array):
        if cp is None:
            raise RuntimeError(_CUDA_STATUS)
        m = 1 << len(targets)
        psi = cp.asarray(self._as_tensor())
        U_gpu = cp.asarray(np.asarray(U, dtype=self.dtype))
        axes = tuple(self._axis_of(q) for q in targets)
        axes_front = tuple(reversed(axes))
        rest_axes = tuple(ax for ax in range(self.num_qubits) if ax not in axes)
        perm = axes_front + rest_axes
        permuted = cp.transpose(psi, perm)
        front = permuted.reshape((m, -1))
        front2 = U_gpu @ front
        reshaped = front2.reshape((2,) * self.num_qubits)
        psi2 = cp.transpose(reshaped, tuple(np.argsort(perm)))
        self.state = cp.asnumpy(psi2.reshape(self.dim)).astype(self.dtype, copy=False)

    def _apply_controlled_unitary_gpu(
        self,
        controls: Tuple[int, ...],
        targets: Tuple[int, ...],
        U: Array,
        ctrl_state: Tuple[int, ...],
    ):
        if cp is None:
            raise RuntimeError(_CUDA_STATUS)
        m = 1 << len(targets)
        psi = cp.asarray(self._as_tensor())
        U_gpu = cp.asarray(np.asarray(U, dtype=self.dtype))
        ax_c = tuple(self._axis_of(q) for q in controls)
        ax_t = tuple(self._axis_of(q) for q in targets)
        ax_t_front = tuple(reversed(ax_t))
        rest = tuple(ax for ax in range(self.num_qubits) if ax not in (*ax_c, *ax_t))
        perm = ax_c + ax_t_front + rest
        moved = cp.transpose(psi, perm)
        idx_ctrl = tuple(int(b) for b in ctrl_state)
        block = moved[idx_ctrl]
        front = block.reshape((m, -1))
        front2 = U_gpu @ front
        moved[idx_ctrl] = front2.reshape(block.shape)
        out = cp.transpose(moved, tuple(np.argsort(perm)))
        self.state = cp.asnumpy(out.reshape(self.dim)).astype(self.dtype, copy=False)

    def apply_unitary(self, targets: Sequence[int], U: Array):
        t = _as_tuple(targets)
        k = len(t)
        if k == 0:
            return
        _validate_qubits(self.num_qubits, t)
        m = 1 << k
        U = np.asarray(U, dtype=self.dtype)
        if U.shape != (m, m):
            raise ValueError(f"U must be {(m, m)} for k={k}, got {U.shape}")

        if self._gpu_enabled():
            try:
                self._apply_unitary_gpu(t, U)
            except Exception as exc:
                raise RuntimeError(f"CuPy backend failed while applying a {k}-qubit unitary") from exc
            self._normalize_()
            return

        psi = self._as_tensor()
        axes = tuple(self._axis_of(q) for q in t)
        axes_front = tuple(reversed(axes))
        rest_axes = tuple(ax for ax in range(self.num_qubits) if ax not in axes)
        perm = axes_front + rest_axes
        permuted = np.transpose(psi, perm)
        front = permuted.reshape((m, -1))
        front2 = U @ front
        reshaped = front2.reshape((2,) * self.num_qubits)
        psi2 = np.transpose(reshaped, np.argsort(perm))
        self.state = psi2.reshape(self.dim).astype(self.dtype, copy=False)
        self._normalize_()

    def apply_controlled_unitary(
        self,
        controls: Sequence[int],
        targets: Sequence[int],
        U: Array,
        ctrl_state: Optional[Sequence[int]] = None,
    ):
        c = _as_tuple(controls)
        t = _as_tuple(targets)
        _validate_qubits(self.num_qubits, (*c, *t))
        if set(c) & set(t):
            raise ValueError("controls and targets must be disjoint")
        if ctrl_state is None:
            ctrl_state = (1,) * len(c)
        ctrl_state = _as_tuple(ctrl_state)
        if len(ctrl_state) != len(c):
            raise ValueError("ctrl_state length must match number of controls")

        k = len(t)
        if k == 0:
            return
        m = 1 << k
        U = np.asarray(U, dtype=self.dtype)
        if U.shape != (m, m):
            raise ValueError(f"U must be {(m, m)} for k={k}, got {U.shape}")

        if self._gpu_enabled():
            try:
                self._apply_controlled_unitary_gpu(c, t, U, ctrl_state)
            except Exception as exc:
                raise RuntimeError("CuPy backend failed while applying a controlled unitary") from exc
            self._normalize_()
            return

        psi = self._as_tensor()
        ax_c = tuple(self._axis_of(q) for q in c)
        ax_t = tuple(self._axis_of(q) for q in t)
        ax_t_front = tuple(reversed(ax_t))
        rest = tuple(ax for ax in range(self.num_qubits) if ax not in (*ax_c, *ax_t))
        perm = ax_c + ax_t_front + rest
        moved = np.transpose(psi, perm)
        idx_ctrl = tuple(int(b) for b in ctrl_state)
        slices = idx_ctrl + (slice(None),) * (len(ax_t) + len(rest))
        block = moved[slices]
        front = block.reshape((m, -1))
        front2 = U @ front
        moved[slices] = front2.reshape(block.shape)
        out = np.transpose(moved, np.argsort(perm))
        self.state = out.reshape(self.dim).astype(self.dtype, copy=False)
        self._normalize_()

    def apply_global_unitary_full(self, U_full: Array):
        U_full = np.asarray(U_full, dtype=self.dtype)
        if U_full.shape != (self.dim, self.dim):
            raise ValueError(f"U_full must be {(self.dim, self.dim)}")
        if self._gpu_enabled():
            try:
                psi = cp.asarray(self.state)
                out = cp.asarray(U_full) @ psi
                self.state = cp.asnumpy(out).astype(self.dtype, copy=False)
            except Exception as exc:
                raise RuntimeError("CuPy backend failed while applying a full unitary") from exc
            self._normalize_()
            return
        self.state = (U_full @ self.state).astype(self.dtype, copy=False)
        self._normalize_()

    def _apply_operator(self, targets: Sequence[int], M: Array) -> Array:
        t = _as_tuple(targets)
        k = len(t)
        if k == 0:
            return self.state.copy()
        _validate_qubits(self.num_qubits, t)
        m = 1 << k
        M = np.asarray(M, dtype=self.dtype)
        if M.shape != (m, m):
            raise ValueError(f"Operator must be {(m, m)} for k={k}")
        psi = self._as_tensor()
        axes = tuple(self._axis_of(q) for q in t)
        axes_front = tuple(reversed(axes))
        rest_axes = tuple(ax for ax in range(self.num_qubits) if ax not in axes)
        perm = axes_front + rest_axes
        moved = np.transpose(psi, perm)
        front = moved.reshape((m, -1))
        front2 = M @ front
        reshaped = front2.reshape((2,) * self.num_qubits)
        psi2 = np.transpose(reshaped, np.argsort(perm))
        return psi2.reshape(self.dim).astype(self.dtype, copy=False)

    def apply_channel(self, targets: Sequence[int], kraus_ops: Sequence[Array]):
        ops = [np.asarray(K, dtype=self.dtype) for K in kraus_ops]
        candidates: List[Array] = []
        probs: List[float] = []
        for K in ops:
            v = self._apply_operator(targets, K)
            p = float(np.vdot(v, v).real)
            candidates.append(v)
            probs.append(max(p, 0.0))
        s = sum(probs)
        if s < _EPS:
            return
        probs = [p / s for p in probs]
        idx = int(self.rng.choice(len(ops), p=probs))
        v = candidates[idx]
        self.state = v / math.sqrt(max(np.vdot(v, v).real, _EPS))

    def noise_bit_flip(self, q: int, p: float):
        p = min(max(float(p), 0.0), 1.0)
        K0 = math.sqrt(max(0.0, 1.0 - p)) * np.eye(2, dtype=self.dtype)
        K1 = math.sqrt(max(0.0, p)) * np.array([[0, 1], [1, 0]], dtype=self.dtype)
        self.apply_channel([q], [K0, K1])

    def noise_phase_flip(self, q: int, p: float):
        p = min(max(float(p), 0.0), 1.0)
        K0 = math.sqrt(max(0.0, 1.0 - p)) * np.eye(2, dtype=self.dtype)
        K1 = math.sqrt(max(0.0, p)) * np.array([[1, 0], [0, -1]], dtype=self.dtype)
        self.apply_channel([q], [K0, K1])

    def noise_depolarizing(self, q: int, p: float):
        p = min(max(float(p), 0.0), 1.0)
        ops = [math.sqrt(max(0.0, 1.0 - p)) * np.eye(2, dtype=self.dtype)]
        if p > 0.0:
            coef = math.sqrt(p / 3.0)
            ops.extend(
                [
                    coef * np.array([[0, 1], [1, 0]], dtype=self.dtype),
                    coef * np.array([[0, -1j], [1j, 0]], dtype=self.dtype),
                    coef * np.array([[1, 0], [0, -1]], dtype=self.dtype),
                ]
            )
        self.apply_channel([q], ops)

    def noise_amplitude_damping(self, q: int, p: float):
        p = min(max(float(p), 0.0), 1.0)
        K0 = np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - p))]], dtype=self.dtype)
        K1 = np.array([[0.0, math.sqrt(max(0.0, p))], [0.0, 0.0]], dtype=self.dtype)
        self.apply_channel([q], [K0, K1])

    def noise_phase_damping(self, q: int, p: float):
        p = min(max(float(p), 0.0), 1.0)
        K0 = np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - p))]], dtype=self.dtype)
        K1 = np.array([[0.0, 0.0], [0.0, math.sqrt(max(0.0, p))]], dtype=self.dtype)
        self.apply_channel([q], [K0, K1])

    @staticmethod
    def Ux(theta: float) -> Array:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def Uy(theta: float) -> Array:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def Uz(theta: float) -> Array:
        a = theta / 2
        return np.array([[np.exp(-1j * a), 0], [0, np.exp(1j * a)]], dtype=np.complex128)

    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> Array:
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
            dtype=np.complex128,
        )

    def I(self, q: int):
        self.apply_unitary([q], np.eye(2, dtype=self.dtype))

    def H(self, q: int):
        inv = 1.0 / math.sqrt(2.0)
        self.apply_unitary([q], np.array([[inv, inv], [inv, -inv]], dtype=self.dtype))

    def S(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, 1j]], dtype=self.dtype))

    def Sdg(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, -1j]], dtype=self.dtype))

    def T(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=self.dtype))

    def Tdg(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=self.dtype))

    def SX(self, q: int):
        self.apply_unitary([q], 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=self.dtype))

    def SXdg(self, q: int):
        self.apply_unitary([q], 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=self.dtype))

    def X(self, q: int):
        self.apply_unitary([q], np.array([[0, 1], [1, 0]], dtype=self.dtype))

    def Y(self, q: int):
        self.apply_unitary([q], np.array([[0, -1j], [1j, 0]], dtype=self.dtype))

    def Z(self, q: int):
        self.apply_unitary([q], np.array([[1, 0], [0, -1]], dtype=self.dtype))

    def P(self, q: int, lam: float):
        self.apply_unitary([q], np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=self.dtype))

    def U1(self, q: int, lam: float):
        self.P(q, lam)

    def U2(self, q: int, phi: float, lam: float):
        self.U(q, math.pi / 2.0, phi, lam)

    def RX(self, q: int, theta: float):
        self.apply_unitary([q], self.Ux(theta))

    def RY(self, q: int, theta: float):
        self.apply_unitary([q], self.Uy(theta))

    def RZ(self, q: int, theta: float):
        self.apply_unitary([q], self.Uz(theta))

    def U(self, q: int, theta: float, phi: float, lam: float):
        self.apply_unitary([q], self.U3(theta, phi, lam))

    def SWAP(self, q1: int, q2: int):
        U = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_unitary([q1, q2], U)

    def ISWAP(self, q1: int, q2: int):
        U = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_unitary([q1, q2], U)

    def ISWAP_theta(self, q1: int, q2: int, theta: float):
        c, s = math.cos(theta), math.sin(theta)
        U = np.array([[1, 0, 0, 0], [0, c, 1j * s, 0], [0, 1j * s, c, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_unitary([q1, q2], U)

    def ISWAP_pow(self, q1: int, q2: int, p: float):
        self.ISWAP_theta(q1, q2, p * math.pi / 2.0)

    def ISWAPdg(self, q1: int, q2: int):
        self.ISWAP_theta(q1, q2, -math.pi / 2.0)

    def fSim(self, q1: int, q2: int, theta: float, phi: float):
        c, s = math.cos(theta), math.sin(theta)
        U = np.array([[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, np.exp(-1j * phi)]], dtype=self.dtype)
        self.apply_unitary([q1, q2], U)

    def SYC(self, q1: int, q2: int):
        self.fSim(q1, q2, math.pi / 2.0, math.pi / 6.0)

    def RXX(self, q1: int, q2: int, theta: float):
        a = theta / 2
        I4 = np.eye(4, dtype=self.dtype)
        XX = np.kron(np.array([[0, 1], [1, 0]], dtype=self.dtype), np.array([[0, 1], [1, 0]], dtype=self.dtype))
        self.apply_unitary([q1, q2], (math.cos(a) * I4) - 1j * math.sin(a) * XX)

    def RYY(self, q1: int, q2: int, theta: float):
        a = theta / 2
        I4 = np.eye(4, dtype=self.dtype)
        Y = np.array([[0, -1j], [1j, 0]], dtype=self.dtype)
        YY = np.kron(Y, Y)
        self.apply_unitary([q1, q2], (math.cos(a) * I4) - 1j * math.sin(a) * YY)

    def RZZ(self, q1: int, q2: int, theta: float):
        a = theta / 2
        e = np.exp(-1j * a)
        ed = np.exp(1j * a)
        self.apply_unitary([q1, q2], np.diag([e, ed, ed, e]).astype(self.dtype))

    @staticmethod
    def _PhasedFSim_matrix(theta: float, zeta: float = 0.0, chi: float = 0.0, gamma: float = 0.0, phi: float = 0.0) -> Array:
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.exp(-1j * (gamma + zeta)) * c, -1j * np.exp(-1j * (gamma - chi)) * s, 0.0],
                [0.0, -1j * np.exp(-1j * (gamma + chi)) * s, np.exp(-1j * (gamma - zeta)) * c, 0.0],
                [0.0, 0.0, 0.0, np.exp(-1j * (2 * gamma + phi))],
            ],
            dtype=np.complex128,
        )

    def PhasedFSim(self, q1: int, q2: int, theta: float, zeta: float = 0.0, chi: float = 0.0, gamma: float = 0.0, phi: float = 0.0):
        self.apply_unitary([q1, q2], self._PhasedFSim_matrix(theta, zeta, chi, gamma, phi))

    def CZ_wave(self, q1: int, q2: int, phi: float, zeta: float = 0.0, chi: float = 0.0, gamma: float = 0.0):
        self.PhasedFSim(q1, q2, 0.0, zeta=zeta, chi=chi, gamma=gamma, phi=phi)

    def PhasedISWAP(self, q1: int, q2: int, theta: float, phase: float):
        rz = lambda a: np.array([[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]], dtype=self.dtype)
        pre = np.kron(rz(phase / 2), rz(-phase / 2))
        post = np.kron(rz(-phase / 2), rz(phase / 2))
        c, s = math.cos(theta), math.sin(theta)
        core = np.array([[1, 0, 0, 0], [0, c, 1j * s, 0], [0, 1j * s, c, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_unitary([q1, q2], pre @ core @ post)

    def CX(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[0, 1], [1, 0]], dtype=self.dtype))

    def CY(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[0, -1j], [1j, 0]], dtype=self.dtype))

    def CZ(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, -1]], dtype=self.dtype))

    def CH(self, c: int, t: int):
        inv = 1.0 / math.sqrt(2.0)
        self.apply_controlled_unitary([c], [t], np.array([[inv, inv], [inv, -inv]], dtype=self.dtype))

    def CS(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, 1j]], dtype=self.dtype))

    def CT(self, c: int, t: int):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=self.dtype))

    def CP(self, c: int, t: int, lam: float):
        self.apply_controlled_unitary([c], [t], np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=self.dtype))

    def CRX(self, c: int, t: int, theta: float):
        self.apply_controlled_unitary([c], [t], self.Ux(theta))

    def CRY(self, c: int, t: int, theta: float):
        self.apply_controlled_unitary([c], [t], self.Uy(theta))

    def CRZ(self, c: int, t: int, theta: float):
        self.apply_controlled_unitary([c], [t], self.Uz(theta))

    def CU(self, c: int, t: int, U: Array):
        U = np.asarray(U, dtype=self.dtype)
        if U.shape != (2, 2):
            raise ValueError("CU expects a 2x2 unitary for the target qubit")
        self.apply_controlled_unitary([c], [t], U)

    def CU1(self, c: int, t: int, lam: float):
        self.CP(c, t, lam)

    def CU3(self, c: int, t: int, theta: float, phi: float, lam: float):
        self.apply_controlled_unitary([c], [t], self.U3(theta, phi, lam))

    def Toffoli(self, c1: int, c2: int, t: int):
        self.apply_controlled_unitary([c1, c2], [t], np.array([[0, 1], [1, 0]], dtype=self.dtype))

    def CSWAP(self, c: int, q1: int, q2: int):
        U = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)
        self.apply_controlled_unitary([c], [q1, q2], U)

    def measure(self, q: int, cbit: Optional[int] = None) -> int:
        ax = self._axis_of(q)
        psi = self._as_tensor()
        moved = np.moveaxis(psi, ax, -1)
        amp0 = moved[..., 0].reshape(-1)
        p0 = float((amp0.conj() * amp0).real.sum())
        p0 = min(max(p0, 0.0), 1.0)
        p1 = max(0.0, 1.0 - p0)
        outcome = int(self.rng.random() < p1)
        if outcome == 0:
            moved[..., 1] = 0.0
        else:
            moved[..., 0] = 0.0
        psi_back = np.moveaxis(moved, -1, ax)
        self.state = psi_back.reshape(self.dim)
        self._normalize_()
        if cbit is not None and 0 <= cbit < len(self.creg):
            self.creg[cbit] = outcome
        return outcome

    def measure_all(self) -> List[int]:
        outcomes = []
        for q in range(self.num_qubits - 1, -1, -1):
            outcomes.append(self.measure(q))
        return list(reversed(outcomes))

    def reset(self, q: int):
        ax = self._axis_of(q)
        psi = self._as_tensor()
        moved = np.moveaxis(psi, ax, -1)
        moved[..., 1] = 0.0
        psi_back = np.moveaxis(moved, -1, ax)
        self.state = psi_back.reshape(self.dim)
        self._normalize_()

    def probs(self) -> Array:
        p = np.abs(self.state) ** 2
        s = float(p.sum())
        return p / (s if s > _EPS else 1.0)

    def state_ket(self, tol: float = 1e-9) -> str:
        out = []
        p = self.probs()
        for i, (amp, pr) in enumerate(zip(self.state, p)):
            if pr > tol:
                out.append(f"({amp.real:+.6f}{amp.imag:+.6f}j)|{i:0{self.num_qubits}b}>")
        return " + ".join(out) if out else "0"


def _context_symbols(ctx: ProgramContext) -> Dict[str, object]:
    symbols: Dict[str, object] = dict(ctx.vars)
    for name, indices in ctx.cregs.items():
        bits = [ctx.sim.creg[idx] for idx in indices]
        symbols[name] = RegisterProxy(bits)
    return symbols


def _resolve_qubit_token(token: str, ctx: ProgramContext) -> List[int]:
    token = token.strip()
    if not token:
        raise ValueError("Expected a qubit operand")

    if re.fullmatch(r"\d+", token):
        return [int(token)]

    if token in ctx.qaliases:
        return list(ctx.qaliases[token])

    match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", token)
    if match is not None:
        name = match.group(1)
        index = _to_int(_eval_expression(match.group(2), _context_symbols(ctx)))
        qubits = ctx.qaliases.get(name, ctx.qregs.get(name))
        if qubits is None:
            raise ValueError(f"Unknown quantum register '{name}'")
        return [qubits[index]]

    if token in ctx.qregs:
        return list(ctx.qregs[token])

    raise ValueError(f"Unknown qubit operand '{token}'")


def _resolve_cbit_token(token: str, ctx: ProgramContext) -> List[int]:
    token = token.strip()
    if not token:
        raise ValueError("Expected a classical operand")

    if re.fullmatch(r"\d+", token):
        return [int(token)]

    match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", token)
    if match is not None:
        name = match.group(1)
        index = _to_int(_eval_expression(match.group(2), _context_symbols(ctx)))
        bits = ctx.cregs.get(name)
        if bits is None:
            raise ValueError(f"Unknown classical register '{name}'")
        return [bits[index]]

    if token in ctx.cregs:
        return list(ctx.cregs[token])

    raise ValueError(f"Unknown classical operand '{token}'")


def _declare_qreg(ctx: ProgramContext, name: str, size: int):
    if size < 1:
        raise ValueError("Quantum register size must be >= 1")
    if not ctx.explicit_qregs:
        ctx.qregs = {}
        ctx.explicit_qregs = True
    if name in ctx.qregs:
        raise ValueError(f"Quantum register '{name}' already exists")
    start = sum(len(bits) for bits in ctx.qregs.values())
    ctx.qregs[name] = list(range(start, start + size))
    ctx.sim.reset_simulation(sum(len(bits) for bits in ctx.qregs.values()), preserve_seed=True)
    if ctx.explicit_cregs:
        total_cbits = sum(len(bits) for bits in ctx.cregs.values())
        ctx.sim.creg = [0] * total_cbits
    else:
        ctx.cregs = {"c": list(range(ctx.sim.num_qubits))}


def _set_default_qreg(ctx: ProgramContext, size: int):
    ctx.explicit_qregs = False
    ctx.qregs = {"q": list(range(size))}
    ctx.sim.reset_simulation(size, preserve_seed=True)
    if ctx.explicit_cregs:
        total_cbits = sum(len(bits) for bits in ctx.cregs.values())
        ctx.sim.creg = [0] * total_cbits
    else:
        ctx.cregs = {"c": list(range(ctx.sim.num_qubits))}


def _declare_creg(ctx: ProgramContext, name: str, size: int):
    if size < 1:
        raise ValueError("Classical register size must be >= 1")
    if not ctx.explicit_cregs:
        ctx.cregs = {}
        ctx.explicit_cregs = True
    if name in ctx.cregs:
        raise ValueError(f"Classical register '{name}' already exists")
    start = sum(len(bits) for bits in ctx.cregs.values())
    ctx.cregs[name] = list(range(start, start + size))
    ctx.sim.creg = [0] * sum(len(bits) for bits in ctx.cregs.values())


def _set_default_creg(ctx: ProgramContext, size: int):
    ctx.explicit_cregs = False
    ctx.cregs = {"c": list(range(size))}
    ctx.sim.creg = [0] * size


def _invoke_broadcast(qubit_groups: Sequence[List[int]], callback):
    if not qubit_groups:
        callback()
        return
    lengths = [len(group) for group in qubit_groups]
    max_len = max(lengths)
    if any(length not in (1, max_len) for length in lengths):
        raise ValueError("Register operands must have matching lengths or be scalar")
    for idx in range(max_len):
        callback(*[group[idx if len(group) > 1 else 0] for group in qubit_groups])


def _invoke_measurement(qgroups: Sequence[List[int]], cgroups: Sequence[List[int]], ctx: ProgramContext):
    lengths = [len(group) for group in (*qgroups, *cgroups)]
    max_len = max(lengths)
    if any(length not in (1, max_len) for length in lengths):
        raise ValueError("Measurement operands must have matching lengths or be scalar")
    for idx in range(max_len):
        q = qgroups[0][idx if len(qgroups[0]) > 1 else 0]
        c = cgroups[0][idx if len(cgroups[0]) > 1 else 0]
        ctx.sim.measure(q, c)
        ctx.saw_measurement = True


def _classical_bitstring(ctx: ProgramContext) -> str:
    return "".join(str(int(bit)) for bit in reversed(ctx.sim.creg))


def _sample_bitstring_from_state(ctx: ProgramContext) -> str:
    clone = QuantumSimulator(ctx.sim.num_qubits, seed=None, prefer_gpu=False)
    clone.state = ctx.sim.state.copy()
    clone.rng = np.random.default_rng(int(ctx.sim.rng.integers(0, 2**63 - 1)))
    bits = clone.measure_all()
    return "".join(str(int(bit)) for bit in reversed(bits))


def _execute_gate_invocation(name: str, param_exprs: Sequence[str], quarg_tokens: Sequence[str], ctx: ProgramContext):
    lower = name.lower()
    qubit_groups = [_resolve_qubit_token(token, ctx) for token in quarg_tokens]
    params = [_eval_expression(expr, _context_symbols(ctx)) for expr in param_exprs]

    if name in ctx.gate_defs:
        gate = ctx.gate_defs[name]
        if len(params) != len(gate.params):
            raise ValueError(f"Gate '{name}' expects {len(gate.params)} params, got {len(params)}")
        if len(qubit_groups) != len(gate.qargs):
            raise ValueError(f"Gate '{name}' expects {len(gate.qargs)} qubit args, got {len(qubit_groups)}")

        def run_custom(*resolved_qubits: int):
            child = ctx.child()
            child.vars.update({param_name: value for param_name, value in zip(gate.params, params)})
            child.qaliases.update({qarg: [q] for qarg, q in zip(gate.qargs, resolved_qubits)})
            _execute_statements(gate.body, child)
            ctx.saw_measurement = ctx.saw_measurement or child.saw_measurement

        _invoke_broadcast(qubit_groups, run_custom)
        return

    single_no_param = {
        "i": "I",
        "id": "I",
        "h": "H",
        "s": "S",
        "sdg": "Sdg",
        "t": "T",
        "tdg": "Tdg",
        "sx": "SX",
        "sxdg": "SXdg",
        "x": "X",
        "y": "Y",
        "z": "Z",
    }
    single_one_param = {"rx": "RX", "ry": "RY", "rz": "RZ", "p": "P", "u1": "U1"}
    double_no_param = {
        "swap": "SWAP",
        "iswap": "ISWAP",
        "iswapdg": "ISWAPdg",
        "syc": "SYC",
        "cx": "CX",
        "cy": "CY",
        "cz": "CZ",
        "ch": "CH",
        "cs": "CS",
        "ct": "CT",
    }
    double_one_param = {
        "iswap_theta": "ISWAP_theta",
        "iswap_pow": "ISWAP_pow",
        "rxx": "RXX",
        "ryy": "RYY",
        "rzz": "RZZ",
        "cp": "CP",
        "crx": "CRX",
        "cry": "CRY",
        "crz": "CRZ",
        "cu1": "CU1",
    }

    if lower in single_no_param:
        if params:
            raise ValueError(f"Gate '{name}' does not take parameters")

        def run_single(q: int):
            getattr(ctx.sim, single_no_param[lower])(q)

        _invoke_broadcast(qubit_groups, run_single)
        return

    if lower in single_one_param:
        if len(params) != 1:
            raise ValueError(f"Gate '{name}' expects 1 parameter")
        theta = _to_float(params[0])

        def run_single(q: int):
            getattr(ctx.sim, single_one_param[lower])(q, theta)

        _invoke_broadcast(qubit_groups, run_single)
        return

    if lower in ("u", "u3"):
        if len(params) != 3:
            raise ValueError(f"Gate '{name}' expects 3 parameters")
        theta, phi, lam = map(_to_float, params)

        def run_u(q: int):
            ctx.sim.U(q, theta, phi, lam)

        _invoke_broadcast(qubit_groups, run_u)
        return

    if lower == "u2":
        if len(params) != 2:
            raise ValueError("u2 expects 2 parameters")
        phi, lam = map(_to_float, params)

        def run_u2(q: int):
            ctx.sim.U2(q, phi, lam)

        _invoke_broadcast(qubit_groups, run_u2)
        return

    if lower in double_no_param:
        if params:
            raise ValueError(f"Gate '{name}' does not take parameters")

        def run_double(q1: int, q2: int):
            getattr(ctx.sim, double_no_param[lower])(q1, q2)

        _invoke_broadcast(qubit_groups, run_double)
        return

    if lower in double_one_param:
        if len(params) != 1:
            raise ValueError(f"Gate '{name}' expects 1 parameter")
        theta = _to_float(params[0])

        def run_double(q1: int, q2: int):
            getattr(ctx.sim, double_one_param[lower])(q1, q2, theta)

        _invoke_broadcast(qubit_groups, run_double)
        return

    if lower == "fsim":
        if len(params) != 2:
            raise ValueError("fsim expects 2 parameters")
        theta, phi = map(_to_float, params)

        def run_fsim(q1: int, q2: int):
            ctx.sim.fSim(q1, q2, theta, phi)

        _invoke_broadcast(qubit_groups, run_fsim)
        return

    if lower == "phased_iswap":
        if len(params) != 2:
            raise ValueError("phased_iswap expects 2 parameters")
        theta, phase = map(_to_float, params)

        def run_phased_iswap(q1: int, q2: int):
            ctx.sim.PhasedISWAP(q1, q2, theta, phase)

        _invoke_broadcast(qubit_groups, run_phased_iswap)
        return

    if lower == "phasedfsim":
        if len(params) != 5:
            raise ValueError("phasedfsim expects 5 parameters")
        theta, zeta, chi, gamma, phi = map(_to_float, params)

        def run_phased_fsim(q1: int, q2: int):
            ctx.sim.PhasedFSim(q1, q2, theta, zeta, chi, gamma, phi)

        _invoke_broadcast(qubit_groups, run_phased_fsim)
        return

    if lower == "cz_wave":
        if len(params) not in (1, 4):
            raise ValueError("cz_wave expects 1 or 4 parameters")
        values = list(map(_to_float, params))
        if len(values) == 1:
            phi, zeta, chi, gamma = values[0], 0.0, 0.0, 0.0
        else:
            phi, zeta, chi, gamma = values

        def run_cz_wave(q1: int, q2: int):
            ctx.sim.CZ_wave(q1, q2, phi, zeta, chi, gamma)

        _invoke_broadcast(qubit_groups, run_cz_wave)
        return

    if lower == "cu3":
        if len(params) != 3:
            raise ValueError("cu3 expects 3 parameters")
        theta, phi, lam = map(_to_float, params)

        def run_cu3(c: int, t: int):
            ctx.sim.CU3(c, t, theta, phi, lam)

        _invoke_broadcast(qubit_groups, run_cu3)
        return

    if lower in ("toffoli", "ccx"):
        if params:
            raise ValueError(f"Gate '{name}' does not take parameters")

        def run_ccx(c1: int, c2: int, target: int):
            ctx.sim.Toffoli(c1, c2, target)

        _invoke_broadcast(qubit_groups, run_ccx)
        return

    if lower == "cswap":
        if params:
            raise ValueError("cswap does not take parameters")

        def run_cswap(control: int, q1: int, q2: int):
            ctx.sim.CSWAP(control, q1, q2)

        _invoke_broadcast(qubit_groups, run_cswap)
        return

    raise ValueError(f"Unknown gate '{name}'")


def _parse_invocation(text: str) -> Tuple[str, List[str], List[str]]:
    match = re.match(r"([A-Za-z_]\w*)", text)
    if match is None:
        raise ValueError(f"Could not parse instruction '{text}'")
    name = match.group(1)
    rest = text[match.end() :].lstrip()
    params: List[str] = []
    if rest.startswith("("):
        inner, rest = _consume_balanced(rest, "(", ")")
        params = [part for part in _split_csv(inner) if part]
        rest = rest.strip()
    args = _split_operands(rest) if rest else []
    return name, params, args


def _execute_legacy_u(text: str, ctx: ProgramContext) -> bool:
    if re.match(r"u\s*\(", text, flags=re.IGNORECASE):
        return False
    parts = text.split()
    if not parts or parts[0].lower() != "u":
        return False
    if len(parts) < 2:
        raise ValueError("u expects a target list")
    target_token = parts[1]
    if "," in target_token:
        targets = []
        for item in target_token.split(","):
            targets.extend(_resolve_qubit_token(item.strip(), ctx))
    else:
        targets = _resolve_qubit_token(target_token, ctx)
    m = 1 << len(targets)
    remaining = parts[2:]
    if len(remaining) == m * m:
        flat = [_parse_complex_qasm(tok) for tok in remaining]
        U = np.array(flat, dtype=np.complex128).reshape(m, m)
        ctx.sim.apply_unitary(targets, U)
        return True
    if len(remaining) == 3 and len(targets) == 1:
        theta, phi, lam = (_to_float(_eval_expression(tok, _context_symbols(ctx))) for tok in remaining)
        ctx.sim.U(targets[0], theta, phi, lam)
        return True
    raise ValueError(f"u expects either 3 parameters or {m * m} matrix entries")


def _execute_command(text: str, ctx: ProgramContext):
    lowered = text.lower().strip()
    if not lowered:
        return

    if _execute_legacy_u(text, ctx):
        return

    if lowered.startswith("openqasm"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("OPENQASM header requires a version")
        ctx.qasm_version = parts[1].strip()
        return

    qreg_match = re.fullmatch(r"qreg\s+([A-Za-z_]\w*)\[(.+)\]", text, flags=re.IGNORECASE)
    if qreg_match is not None:
        _declare_qreg(ctx, qreg_match.group(1), _to_int(_eval_expression(qreg_match.group(2), _context_symbols(ctx))))
        return

    if re.fullmatch(r"qreg\s+\d+", text, flags=re.IGNORECASE):
        _set_default_qreg(ctx, int(text.split()[1]))
        return

    creg_match = re.fullmatch(r"creg\s+([A-Za-z_]\w*)\[(.+)\]", text, flags=re.IGNORECASE)
    if creg_match is not None:
        _declare_creg(ctx, creg_match.group(1), _to_int(_eval_expression(creg_match.group(2), _context_symbols(ctx))))
        return

    if re.fullmatch(r"creg\s+\d+", text, flags=re.IGNORECASE):
        _set_default_creg(ctx, int(text.split()[1]))
        return

    qubit_array = re.fullmatch(r"qubit\[(.+)\]\s+([A-Za-z_]\w*)", text, flags=re.IGNORECASE)
    if qubit_array is not None:
        _declare_qreg(ctx, qubit_array.group(2), _to_int(_eval_expression(qubit_array.group(1), _context_symbols(ctx))))
        return

    if re.fullmatch(r"qubit\s+[A-Za-z_]\w*", text, flags=re.IGNORECASE):
        _declare_qreg(ctx, text.split()[1], 1)
        return

    bit_array = re.fullmatch(r"bit\[(.+)\]\s+([A-Za-z_]\w*)", text, flags=re.IGNORECASE)
    if bit_array is not None:
        _declare_creg(ctx, bit_array.group(2), _to_int(_eval_expression(bit_array.group(1), _context_symbols(ctx))))
        return

    if re.fullmatch(r"bit\s+[A-Za-z_]\w*", text, flags=re.IGNORECASE):
        _declare_creg(ctx, text.split()[1], 1)
        return

    if lowered.startswith("shots "):
        return

    if lowered.startswith("seed "):
        seed_expr = text.split(maxsplit=1)[1]
        ctx.sim.reseed(_to_int(_eval_expression(seed_expr, _context_symbols(ctx))))
        return

    if lowered.startswith("u_full "):
        parts = text.split()[1:]
        N = 1 << ctx.sim.num_qubits
        if len(parts) != N * N:
            raise ValueError(f"u_full expects {N * N} entries for N={N}")
        flat = [_parse_complex_qasm(tok) for tok in parts]
        ctx.sim.apply_global_unitary_full(np.array(flat, dtype=np.complex128).reshape(N, N))
        return

    if lowered.startswith("measure "):
        measure_match = re.fullmatch(r"measure\s+(.+?)\s*->\s*(.+)", text, flags=re.IGNORECASE)
        if measure_match is not None:
            qgroups = [_resolve_qubit_token(measure_match.group(1).strip(), ctx)]
            cgroups = [_resolve_cbit_token(measure_match.group(2).strip(), ctx)]
            _invoke_measurement(qgroups, cgroups, ctx)
            return

        parts = _split_operands(text)
        if len(parts) not in (2, 3):
            raise ValueError("measure expects 'measure q' or 'measure q cbit'")
        qgroup = _resolve_qubit_token(parts[1], ctx)
        if len(parts) == 2:
            for q in qgroup:
                ctx.sim.measure(q)
                ctx.saw_measurement = True
            return
        cgroup = _resolve_cbit_token(parts[2], ctx)
        _invoke_measurement([qgroup], [cgroup], ctx)
        return

    if lowered.startswith("reset "):
        parts = _split_operands(text)
        if len(parts) != 2:
            raise ValueError("reset expects one operand")
        for q in _resolve_qubit_token(parts[1], ctx):
            ctx.sim.reset(q)
        return

    if lowered == "print_state":
        print(ctx.sim.state_ket())
        return

    if lowered == "print_probs":
        probs = ctx.sim.probs()
        for i, pr in enumerate(probs):
            if pr > 1e-12:
                print(f"|{i:0{ctx.sim.num_qubits}b}>: {pr:.6f}")
        return

    if lowered == "print_creg":
        for i, bit in enumerate(ctx.sim.creg):
            print(f"c[{i}] = {bit}")
        return

    if lowered in ("barrier", "delay"):
        return

    parts = _split_operands(text)
    if not parts:
        return
    cmd = parts[0].lower()
    if cmd in ("noise_bitflip", "nbf", "noise_phaseflip", "npf", "noise_depolarizing", "ndp", "noise_amp", "nad", "noise_phase", "nph"):
        if len(parts) != 3:
            raise ValueError(f"{parts[0]} expects a target and a probability")
        qgroup = _resolve_qubit_token(parts[1], ctx)
        p = _to_float(_eval_expression(parts[2], _context_symbols(ctx)))
        method_name = {
            "noise_bitflip": "noise_bit_flip",
            "nbf": "noise_bit_flip",
            "noise_phaseflip": "noise_phase_flip",
            "npf": "noise_phase_flip",
            "noise_depolarizing": "noise_depolarizing",
            "ndp": "noise_depolarizing",
            "noise_amp": "noise_amplitude_damping",
            "nad": "noise_amplitude_damping",
            "noise_phase": "noise_phase_damping",
            "nph": "noise_phase_damping",
        }[cmd]

        def run_noise(q: int):
            getattr(ctx.sim, method_name)(q, p)

        _invoke_broadcast([qgroup], run_noise)
        return

    name, params, args = _parse_invocation(text)
    if not args and name.lower() not in ("print_state", "print_probs", "print_creg"):
        raise ValueError(f"Instruction '{text}' is missing operands")
    _execute_gate_invocation(name, params, args, ctx)


def _iterable_values(spec: str, ctx: ProgramContext) -> List[int]:
    spec = spec.strip()
    symbols = _context_symbols(ctx)
    if spec.startswith("[") and spec.endswith("]"):
        parts = [part.strip() for part in _split_top_level(spec[1:-1], split_on_whitespace=False) if part.strip()]
        if len(parts) == 1 and ":" in parts[0]:
            parts = [part.strip() for part in parts[0].split(":")]
        if len(parts) == 2:
            start = _to_int(_eval_expression(parts[0], symbols))
            end = _to_int(_eval_expression(parts[1], symbols))
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
        if len(parts) == 3:
            start = _to_int(_eval_expression(parts[0], symbols))
            step = _to_int(_eval_expression(parts[1], symbols))
            end = _to_int(_eval_expression(parts[2], symbols))
            if step == 0:
                raise ValueError("Loop step cannot be zero")
            stop = end + (1 if step > 0 else -1)
            return list(range(start, stop, step))
        raise ValueError(f"Unsupported loop range '{spec}'")
    if spec.startswith("{") and spec.endswith("}"):
        return [_to_int(_eval_expression(item, symbols)) for item in _split_csv(spec[1:-1])]
    if spec.lower().startswith("range(") and spec.endswith(")"):
        args = [_to_int(_eval_expression(item, symbols)) for item in _split_csv(spec[6:-1])]
        return list(range(*args))
    value = _eval_expression(spec, symbols)
    if isinstance(value, (list, tuple)):
        return [_to_int(v) for v in value]
    raise ValueError(f"Unsupported iterable '{spec}'")


def _execute_statements(statements: Sequence[Statement], ctx: ProgramContext):
    for stmt in statements:
        if stmt.kind == "cmd":
            _execute_command(stmt.text, ctx)
            continue
        if stmt.kind == "gate":
            ctx.gate_defs[stmt.name] = GateDefinition(stmt.name, stmt.params, stmt.qargs, stmt.body)
            continue
        if stmt.kind == "if":
            branch = stmt.body if bool(_eval_expression(stmt.condition, _context_symbols(ctx))) else stmt.else_body
            _execute_statements(branch, ctx)
            continue
        if stmt.kind == "for":
            values = _iterable_values(stmt.iterable, ctx)
            for value in values:
                previous = ctx.vars.get(stmt.var_name)
                had_previous = stmt.var_name in ctx.vars
                ctx.vars[stmt.var_name] = value
                _execute_statements(stmt.body, ctx)
                if had_previous:
                    ctx.vars[stmt.var_name] = previous
                else:
                    ctx.vars.pop(stmt.var_name, None)
            continue
        if stmt.kind == "while":
            iterations = 0
            while bool(_eval_expression(stmt.condition, _context_symbols(ctx))):
                iterations += 1
                if iterations > _MAX_WHILE_ITERATIONS:
                    raise RuntimeError("while loop exceeded the safety iteration limit")
                _execute_statements(stmt.body, ctx)
            continue
        raise ValueError(f"Unknown statement kind '{stmt.kind}'")


def _extract_shots(statements: Sequence[Statement]) -> Tuple[int, List[Statement]]:
    shots = 1
    remaining: List[Statement] = []
    for stmt in statements:
        if stmt.kind == "cmd":
            parts = _split_operands(stmt.text)
            if parts and parts[0].lower() == "shots":
                if len(parts) != 2:
                    raise ValueError("shots expects exactly one integer expression")
                shots = _to_int(_eval_expression(parts[1], {"pi": math.pi, "tau": math.tau, "e": math.e}))
                continue
        remaining.append(stmt)
    if shots < 1:
        raise ValueError("shots must be >= 1")
    return shots, remaining


def execute_qasm(sim: QuantumSimulator, lines: Optional[List[str]] = None, base_path: Optional[Path] = None):
    """Execute the local OpenQASM-like program on ``sim``."""
    if lines is None:
        print("Enter QASM commands; type 'run' to execute:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "run":
                break
            lines.append(line)

    program_text = ";\n".join(lines)
    base = Path(base_path) if base_path is not None else None
    statements = _parse_program(program_text, base)
    shots, statements = _extract_shots(statements)

    if shots == 1:
        ctx = ProgramContext(sim=sim, base_path=base, qregs={"q": list(range(sim.num_qubits))}, cregs={"c": list(range(len(sim.creg)))})
        _execute_statements(statements, ctx)
        print(sim.state_ket())
        for i, bit in enumerate(sim.creg):
            print(f"c[{i}] = {bit}")
        return {"shots": 1, "counts": None, "state": sim.state.copy(), "creg": list(sim.creg), "backend": sim.backend_name}

    counts: Counter[str] = Counter()
    seed_rng = np.random.default_rng(sim.seed) if sim.seed is not None else None
    last_ctx: Optional[ProgramContext] = None
    initial_qubits = max(sim.num_qubits, 1)

    for _ in range(shots):
        shot_seed = None if seed_rng is None else int(seed_rng.integers(0, 2**63 - 1))
        shot_sim = QuantumSimulator(initial_qubits, seed=shot_seed, prefer_gpu=sim.prefer_gpu)
        ctx = ProgramContext(sim=shot_sim, base_path=base, qregs={"q": list(range(shot_sim.num_qubits))}, cregs={"c": list(range(len(shot_sim.creg)))})
        _execute_statements(statements, ctx)
        bitstring = _classical_bitstring(ctx) if ctx.saw_measurement else _sample_bitstring_from_state(ctx)
        counts[bitstring] += 1
        last_ctx = ctx

    if last_ctx is not None:
        sim.copy_from(last_ctx.sim)

    print(f"shots = {shots}")
    for bitstring in sorted(counts):
        print(f"{bitstring}: {counts[bitstring]}")
    return {"shots": shots, "counts": dict(counts), "state": sim.state.copy(), "creg": list(sim.creg), "backend": sim.backend_name}
