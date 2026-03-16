"""
Calculator skill — safe arithmetic expression evaluator.

Uses Python's ast module so no arbitrary code can be executed.
Supports: + - * / ** % // and parentheses.
"""
import ast
import operator

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported literal type: {type(node.value).__name__}")
        return node.value
    elif isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _OPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        return _OPS[op](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _OPS:
            raise ValueError(f"Unsupported unary operator: {op.__name__}")
        return _OPS[op](_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


@tool(parse_docstring=True)
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression and return the result.
    Supports +, -, *, /, ** (power), % (modulo), // (floor division), and parentheses.

    Args:
        expression: A math expression such as '2 + 2', '(10 * 5) / 2', or '2 ** 10'.

    Returns:
        string: The expression and its result, e.g. '2 ** 10 = 1024'.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree.body)
        # Format cleanly: show int if result is a whole number
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: division by zero in '{expression}'"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def register() -> dict:
    return {
        "calculate": (calculate, "Evaluate a mathematical expression (+, -, *, /, **, %, //)."),
    }
