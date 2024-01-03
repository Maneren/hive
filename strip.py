from __future__ import annotations

import ast
import pathlib
import sys

import astunparse

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input file> [<output file>]")
    sys.exit(1)

input_file_path = sys.argv[1]

input_file = pathlib.Path(input_file_path).read_text()


class TypeHintRemover(ast.NodeTransformer):
    # remove type annotations and docstrings from functions
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        self.generic_visit(node)

        node.returns = None

        if not node.body:
            return node

        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and hasattr(first, "value")
            and isinstance(
                first.value,
                ast.Constant,
            )
        ):
            node.body = node.body[1:]

        return node

    # remove type annotations from args
    def visit_arg(self, node: ast.arg) -> ast.AST | None:
        self.generic_visit(node)

        node.annotation = None
        return node

    # remove type annotations, docstrings from classes and handle dataclasses
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST | None:
        # remove all docstrings
        if not node.body:
            return node

        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and hasattr(first, "value")
            and isinstance(
                first.value,
                ast.Constant,
            )
        ):
            node.body = node.body[1:]

        # remove and collect all class attributes
        class_vars = []
        to_pop = []
        for i, subnode in enumerate(node.body):
            if isinstance(subnode, ast.AnnAssign):
                class_vars.append(subnode)
                to_pop.append(i)

        for i in reversed(to_pop):
            node.body.pop(i)

        decorators = [
            decorator.id
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Name)
        ]

        if "dataclass" in decorators:
            node.decorator_list = []
            self.transform_dataclass(class_vars, node)

        self.generic_visit(node)

        return node

    # manually implement simplified dataclass decorator
    def transform_dataclass(
        self,
        class_vars: list[ast.AnnAssign],
        node: ast.ClassDef,
    ) -> None:
        arguments = [ast.arg(arg="self")]
        body = []

        for var in class_vars:
            name = var.target.id
            arguments.append(ast.arg(arg=name))
            body.append(
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id="self"), attr=name)],
                    value=ast.Name(id=name),
                ),
            )

        init = ast.FunctionDef(
            name="__init__",
            args=ast.arguments(
                args=arguments,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            annotations=[],
        )

        node.body.insert(0, init)

    # remove all TypeVars and TypeAliases
    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        value = node.value

        if (
            isinstance(value, ast.BinOp)
            and (left := value.left)
            and isinstance(left, ast.Subscript)
        ):
            value = left

        if (
            isinstance(value, ast.Subscript)
            and (subvalue := value.value)
            and isinstance(subvalue, ast.Name)
            and subvalue.id in ("list", "dict", "tuple")
        ):
            return None

        if (
            isinstance(value, ast.Call)
            and (func := value.func)
            and isinstance(func, ast.Name)
            and func.id == "TypeVar"
        ):
            return None

        return node

    # remove all type annotations from assignments
    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if node.value is None:
            return None

        new = ast.Assign([node.target], node.value)
        return self.visit_Assign(new)

    # remove all import 'typing' statements
    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        node.names = [n for n in node.names if n.name != "typing"]
        return node if node.names else None

    # remove all import from 'typing' and '__future__' statements
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        return node if node.module not in {"typing", "__future__"} else None

    # remove all assert statements
    def visit_Assert(self, _node: ast.Assert) -> ast.AST | None:
        return None


# parse the source code into an AST
parsed_source = ast.parse(input_file)
# apply the visitor
transformed = TypeHintRemover().visit(parsed_source)
# convert the AST back to source code
unparsed = astunparse.unparse(transformed)


if len(sys.argv) < 3 or (output_file_path := sys.argv[2]) == input_file_path:
    print(unparsed)
    sys.exit(0)

pathlib.Path(output_file_path).write_text(unparsed)
