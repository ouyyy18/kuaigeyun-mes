#!/usr/bin/env python3
"""Build a code-only graph and add instance-method call edges.

CI entry point for the graphify toolchain. It extracts code with graphify's
AST pass, applies the Python + TypeScript instance-call post-passes, then
re-clusters and exports graph.json. Semantic (doc/paper/image) extraction is
skipped to keep CI deterministic and cheap.

Usage:
  python3 build_code_graph.py --repo . [--out graphify-out] \
      [--typescript .graphify-tools/node_modules/typescript]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="graphify-out")
    ap.add_argument("--typescript", default=".graphify-tools/node_modules/typescript")
    args = ap.parse_args()

    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.detect import detect
    from graphify.export import to_json
    from graphify.extract import collect_files, extract

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "graph.json"

    detection = detect(repo)
    code_files: list[Path] = []
    for item in detection.get("files", {}).get("code", []):
        p = Path(item)
        code_files.extend(collect_files(p) if p.is_dir() else [p])
    if not code_files:
        print("no code files detected", file=sys.stderr)
        return 1

    print(f"extracting {len(code_files)} code files", flush=True)
    started = time.monotonic()
    extraction = extract(code_files, cache_root=repo, root=repo)
    print(f"extracted in {time.monotonic() - started:.1f}s; building graph", flush=True)
    started = time.monotonic()
    G = build_from_json(extraction, root=repo, directed=False)
    if G.number_of_nodes() == 0:
        print("empty graph", file=sys.stderr)
        return 1
    print(
        f"built {G.number_of_nodes()} nodes / {G.number_of_edges()} edges "
        f"in {time.monotonic() - started:.1f}s; writing raw graph",
        flush=True,
    )
    raw_nodes = []
    for nid, data in G.nodes(data=True):
        node = dict(data)
        node["id"] = nid
        raw_nodes.append(node)
    raw_links = []
    for source, target, data in G.edges(data=True):
        edge = dict(data)
        edge["source"] = source
        edge["target"] = target
        raw_links.append(edge)
    graph_path.write_text(
        json.dumps(
            {"nodes": raw_nodes, "links": raw_links, "hyperedges": extraction.get("hyperedges", [])},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"raw graph written: {graph_path}", flush=True)

    run([
        sys.executable,
        str(TOOL_DIR / "instance_call_edges.py"),
        "--repo", str(repo),
        "--graph", str(graph_path),
        "--write",
    ])
    ts_dir = (repo / args.typescript) if not Path(args.typescript).is_absolute() else Path(args.typescript)
    if (ts_dir / "lib" / "typescript.js").exists():
        run([
            "node",
            str(TOOL_DIR / "ts_instance_call_edges.mjs"),
            "--repo", str(repo),
            "--graph", str(graph_path),
            "--typescript", str(ts_dir),
            "--write",
        ])
    else:
        print(f"typescript package not found at {ts_dir}; skipping TS pass", file=sys.stderr)

    patched = json.loads(graph_path.read_text(encoding="utf-8"))
    extraction = {
        "nodes": patched["nodes"],
        "edges": patched["links"],
        "hyperedges": patched.get("hyperedges", []),
        "input_tokens": 0,
        "output_tokens": 0,
    }
    G = build_from_json(extraction, root=repo, directed=False)
    communities = cluster(G)
    to_json(G, communities, str(graph_path))
    postpass = sum(
        1
        for _, _, data in G.edges(data=True)
        if str(data.get("context", "")).endswith("postpass")
    )
    print(
        f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"{len(communities)} communities, {postpass} post-pass edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
