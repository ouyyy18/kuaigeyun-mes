#!/usr/bin/env python3
"""Add EXTRACTED `calls` edges for Python instance-method calls.

graphify's AST pass resolves `ClassName.method()` and typed locals, but not
module-level singletons such as `service = Service()` used as
`service.method(...)`. This deterministic post-pass parses the repository
with Python's `ast` and connects those calls to the class-qualified method
nodes already present in graphify-out/graph.json.

Usage:
  python3 instance_call_edges.py --repo kuaigeyun-mes [--graph graphify-out/graph.json] [--write]

Only edges whose caller, callee, and receiver class all resolve in the graph
are added; ambiguous class names are skipped. New edges carry
`context=instance_call_postpass` and confidence EXTRACTED 1.0.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

METHOD_LABEL = re.compile(r"^\.(.+)\(\)$")
FUNCTION_LABEL = re.compile(r"^([^(]+)\(\)$")


def label_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()


class GraphIndex:
    def __init__(self, data: dict, repo: Path):
        self.data = data
        self.repo = repo
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.links = data["links"]
        self.pairs = {(e.get("source"), e.get("target")) for e in self.links}
        self.file_node: dict[str, str] = {}
        self.function_by_file: dict[str, list[str]] = defaultdict(list)
        self.method_by_node: dict[str, str] = {}
        self.class_methods: dict[str, dict[str, str]] = defaultdict(dict)
        self.class_by_name: dict[str, list[str]] = defaultdict(list)
        self.class_file: dict[str, str] = {}
        self.node_file: dict[str, str] = {}

        for nid, n in self.nodes.items():
            sf = n.get("source_file") or ""
            if not sf:
                continue
            self.node_file[nid] = sf
            if n.get("file_type") == "code" and n.get("label") in (Path(sf).name, Path(sf).stem):
                self.file_node.setdefault(sf, nid)
            lab = n.get("label") or ""
            if lab.endswith("()") and (n.get("file_type") == "code"):
                self.function_by_file[sf].append(nid)
                if lab.startswith("."):
                    self.method_by_node[nid] = lab[1:-2]
                m = FUNCTION_LABEL.match(lab)
                if m and not lab.startswith("."):
                    pass
            if n.get("_callable_class"):
                self.class_by_name[label_key(lab)].append(nid)
                self.class_file[nid] = sf

        # class -> methods via `method` edges (authoritative in graphify)
        for e in self.links:
            if e.get("relation") == "method":
                src, tgt = e.get("source"), e.get("target")
                if src in self.nodes and tgt in self.method_by_node:
                    self.class_methods[src][label_key(self.method_by_node[tgt])] = tgt

        # imports per file node: target ids
        self.imported_by_file: dict[str, set[str]] = defaultdict(set)
        for e in self.links:
            if e.get("relation") in ("imports", "imports_from"):
                src = e.get("source")
                if src in self.file_node.values() or (src in self.nodes and (self.nodes[src].get("label") or "").endswith(Path(self.nodes[src].get("source_file") or "").name)):
                    self.imported_by_file[src].add(e.get("target"))

    def find_caller(self, sf: str, def_line: int, name: str) -> str | None:
        wanted = set()
        for nid in self.function_by_file.get(sf, []):
            loc = self.nodes[nid].get("source_location") or ""
            m = re.match(r"L(\d+)", loc)
            if m and int(m.group(1)) == def_line:
                wanted.add(nid)
        if len(wanted) == 1:
            return next(iter(wanted))
        # fall back to label match
        lab1, lab2 = f".{name}()", f"{name}()"
        cands = [nid for nid in self.function_by_file.get(sf, []) if self.nodes[nid].get("label") in (lab1, lab2)]
        return cands[0] if len(cands) == 1 else None


    def add(self, caller: str, target: str, sf: str, line: int, note: str) -> bool:
        if (caller, target) in self.pairs:
            return False
        self.links.append({
            "source": caller,
            "target": target,
            "relation": "calls",
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "source_file": sf,
            "source_location": f"L{line}",
            "context": "instance_call_postpass",
            "weight": 1.0,
            "_origin": "ast",
        })
        self.pairs.add((caller, target))
        return True


    def resolve_class(self, sf: str, name: str, imported_names: set[str]) -> str | None:
        key = label_key(name)
        imported_keys = {label_key(x) for x in imported_names}
        cands = self.class_by_name.get(key, [])
        if len(cands) == 1:
            cid = cands[0]
            if self.class_file.get(cid) == sf or key in imported_keys:
                return cid
            return None
        local = [c for c in cands if self.class_file.get(c) == sf]
        if len(local) == 1:
            return local[0]
        imported = [c for c in cands
                    if label_key(self.nodes[c].get("label", "")) in imported_keys]
        return imported[0] if len(imported) == 1 else None
def imported_class_names(tree: ast.Module, repo: Path, sf: str) -> set[str]:
    """Class identifiers imported into this module (from-imports and modules)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for a in node.names:
                    names.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
    return names


def class_bindings(body: list[ast.stmt], source: str) -> dict[str, str]:
    """name -> ClassName for `name = Class(...)` at module or function scope."""
    out: dict[str, str] = {}
    for node in body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Name) and f.id[:1].isupper():
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = f.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Name) and f.id[:1].isupper() and isinstance(node.target, ast.Name):
                out[node.target.id] = f.id
    return out


def self_attr_bindings(cls: ast.ClassDef) -> dict[str, str]:
    """field -> ClassName for `self.field = Class(...)` across the class body."""
    out: dict[str, str] = {}
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            if not (isinstance(f, ast.Name) and f.id[:1].isupper()):
                continue
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    out[t.attr] = f.id
    return out


def enclosing(nodes: list[ast.AST], line: int) -> ast.AST | None:
    best = None
    best_span = None
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(n, "lineno", 10**9)
            end = getattr(n, "end_lineno", start)
            if start <= line <= end:
                span = end - start
                if best_span is None or span < best_span:
                    best, best_span = n, span
    return best


def process_file(path: Path, repo: Path, idx: GraphIndex) -> tuple[int, list[dict]]:
    rel = str(path.relative_to(repo))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return 0, []
    imported = imported_class_names(tree, repo, rel)
    module_bindings = class_bindings(tree.body, rel)
    added = 0
    samples: list[dict] = []
    all_nodes = list(ast.walk(tree))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    def caller_for(line: int) -> tuple[str | None, str | None]:
        enc = enclosing(all_nodes, line)
        if isinstance(enc, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return idx.find_caller(rel, enc.lineno, enc.name), enc.name
        return None, None

    def try_call(receiver_class: str | None, receiver: str, method: str, line: int, caller_ctx=None):
        nonlocal added
        if not receiver_class:
            return
        cid = idx.resolve_class(rel, receiver_class, imported)
        if not cid:
            return
        target = idx.class_methods.get(cid, {}).get(label_key(method))
        if not target:
            return
        caller = caller_ctx or caller_for(line)[0]
        if not caller or not idx.add(caller, target, rel, line, f"{receiver}.{method}"):
            return
        added += 1
        if len(samples) < 200:
            samples.append({
                "caller": idx.nodes[caller].get("label"),
                "target": idx.nodes[target].get("label"),
                "class": idx.nodes[cid].get("label"),
                "line": line,
                "source_file": rel,
            })

    # 1) module-level singleton calls inside functions
    for fn in (n for n in all_nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        local = dict(module_bindings)
        local.update(class_bindings(fn.body, rel))
        if local:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    v = node.func.value
                    if isinstance(v, ast.Name) and v.id in local:
                        try_call(local[v.id], v.id, node.func.attr, node.lineno,
                                 idx.find_caller(rel, fn.lineno, fn.name))

    # 2) `self.field.method()` using class-body bindings
    for cname, cls in classes.items():
        field_types = self_attr_bindings(cls)
        if not field_types:
            continue
        for fn in (n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    v = node.func.value
                    if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                            and v.value.id == "self" and v.attr in field_types):
                        try_call(field_types[v.attr], f"self.{v.attr}", node.func.attr,
                                 node.lineno, idx.find_caller(rel, fn.lineno, fn.name))
    return added, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--graph", default="graphify-out/graph.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    graph_path = Path(args.graph)
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    idx = GraphIndex(data, repo)
    total = 0
    all_samples: list[dict] = []
    py_files = sorted(repo.rglob("*.py"))
    for p in py_files:
        if "/.git/" in str(p) or "/node_modules/" in str(p):
            continue
        n, samples = process_file(p, repo, idx)
        total += n
        all_samples.extend(samples[:5])
        if len(all_samples) >= 15:
            all_samples = all_samples[:15]
    print(f"python files: {len(py_files)} | new instance-call edges: {total}")
    for s in all_samples:
        print("  ", s["caller"], "->", s["target"], "|", s["class"], "|", s["source_file"], f"L{s['line']}")
    if args.write:
        graph_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {graph_path}")
    else:
        print("dry run (pass --write to update graph.json)")


if __name__ == "__main__":
    main()
