#!/usr/bin/env node
/**
 * Add EXTRACTED `calls` edges for TypeScript/TSX member calls that graphify's
 * heuristic AST pass leaves unresolved (imported singletons, object literals,
 * hook return values, class instances without a visible constructor binding).
 *
 * Uses the TypeScript checker, so every added edge is an exact symbol
 * resolution to a declaration in this repository - no name heuristics.
 * Object-literal function properties become method nodes when the extractor
 * did not create one, so the call keeps method-level granularity.
 *
 * Usage:
 *   node ts_instance_call_edges.mjs --repo ../kuaigeyun-mes \
 *     --graph graphify-out/graph.json --typescript <ts-package-dir> [--write]
 */
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const k = a.slice(2);
      const v = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : true;
      out[k] = v;
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const repo = path.resolve(args.repo || ".");
const graphPath = path.resolve(args.graph || "graphify-out/graph.json");
const tsPkg = args.typescript
  ? path.resolve(args.typescript)
  : path.resolve("graphify-out/.ts-tools/node_modules/typescript");
const write = Boolean(args.write);

const ts = (await import(path.join(tsPkg, "lib", "typescript.js"))).default;

const graph = JSON.parse(fs.readFileSync(graphPath, "utf8"));
const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
const links = graph.links;
const pairs = new Set(links.map((e) => `${e.source}\u0000${e.target}`));
const newNodeIds = new Set();

const norm = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
const key = (s) => norm(s);
const methodLabel = /^\.?(.+)\(\)$/;

const fileNodeByPath = new Map();
const functionsByFile = new Map();
const classesByFile = new Map();
const methodsByClass = new Map();
const nodeIds = new Set(nodes.keys());

for (const n of graph.nodes) {
  const sf = n.source_file || "";
  if (!sf) continue;
  const base = path.basename(sf);
  if (n.file_type === "code" && (n.label === base || n.label === base.replace(/\.[^.]+$/, ""))) {
    if (!fileNodeByPath.has(sf)) fileNodeByPath.set(sf, n.id);
  }
  if (n.file_type !== "code") continue;
  if ((n.label || "").endsWith("()")) {
    const arr = functionsByFile.get(sf) || [];
    arr.push(n);
    functionsByFile.set(sf, arr);
  }
  if (n._callable_class) {
    const arr = classesByFile.get(sf) || [];
    arr.push(n);
    classesByFile.set(sf, arr);
  }
}
for (const e of links) {
  if (e.relation !== "method") continue;
  const t = nodes.get(e.target);
  const m = t && methodLabel.exec(t.label || "");
  if (!m) continue;
  const mm = methodsByClass.get(e.source) || new Map();
  mm.set(key(m[1]), e.target);
  methodsByClass.set(e.source, mm);
}

function lineOf(n) {
  const m = /^L(\d+)/.exec(n.source_location || "");
  return m ? Number(m[1]) : null;
}

function findNodeAt(sf, line, name) {
  const arr = functionsByFile.get(sf) || [];
  const exact = arr.filter((n) => lineOf(n) === line);
  if (exact.length === 1) return exact[0].id;
  if (name) {
    const byName = arr.filter((n) => key(n.label) === key(name) + "()" || key(n.label) === key(name));
    if (byName.length === 1) return byName[0].id;
  }
  const near = arr.filter((n) => {
    const l = lineOf(n);
    return l != null && Math.abs(l - line) <= 2;
  });
  return near.length === 1 ? near[0].id : null;
}

function relOf(sf) {
  return path.relative(repo, typeof sf === "string" ? sf : sf.fileName);
}

function lineAt(sf, pos) {
  return sf.getLineAndCharacterOfPosition(pos).line + 1;
}

function newIdFor(rel, container, prop) {
  const stem = rel.replace(/\.[^.]+$/, "").replace(/[/.\\-]+/g, "_").toLowerCase().replace(/[^a-z0-9_]/g, "");
  let id = `${stem}_${norm(container)}_${norm(prop)}`;
  let i = 2;
  while (nodeIds.has(id) || newNodeIds.has(id)) id = `${stem}_${norm(container)}_${norm(prop)}_${i++}`;
  return id;
}

function ensureMethodNode(rel, sf, decl, containerName, containerNode) {
  const propName = decl.name && ts.isIdentifier(decl.name) ? decl.name.text : null;
  if (!propName) return null;
  const line = lineAt(sf, decl.getStart());
  const existing = (functionsByFile.get(rel) || []).find(
    (n) => lineOf(n) === line && (key(n.label) === key(propName) + "()" || key(n.label) === key(propName)),
  );
  if (existing) return existing.id;
  const id = newIdFor(rel, containerName, propName);
  const node = {
    id,
    label: `${propName}()`,
    file_type: "code",
    source_file: rel,
    source_location: `L${line}`,
    source_url: null,
    captured_at: null,
    author: null,
    contributor: null,
    _callable: true,
    _callable_class: false,
    _origin: "ast",
  };
  nodes.set(id, node);
  newNodeIds.add(id);
  const parent = containerNode;
  if (!parent) return null;
  graph.links.push({
    source: parent,
    target: id,
    relation: (nodes.get(parent)._callable_class ? "method" : "contains"),
    confidence: "EXTRACTED",
    confidence_score: 1.0,
    source_file: rel,
    source_location: `L${line}`,
    context: "ts_instance_call_postpass",
    weight: 1.0,
    _origin: "ast",
  });
  graph.nodes.push(node);
  const arr = functionsByFile.get(rel) || [];
  arr.push(node);
  functionsByFile.set(rel, arr);
  return id;
}

function findProvider(decl, objLiteral) {
  // Nearest enclosing function/method that owns an object literal return or
  // variable; fall back to the object's variable/class-field container.
  let n = objLiteral ? objLiteral.parent : decl.parent;
  let containerName = null;
  while (n) {
    if (ts.isVariableDeclaration(n) && n.name && ts.isIdentifier(n.name)) {
      containerName = n.name.text;
      const rel = relOf(n.getSourceFile());
      const fn = findNodeAt(rel, lineAt(n.getSourceFile(), n.getStart()), n.name.text);
      if (fn) return { node: fn, name: containerName };
    }
    if (
      ts.isFunctionDeclaration(n) ||
      ts.isMethodDeclaration(n) ||
      ts.isArrowFunction(n) ||
      ts.isFunctionExpression(n)
    ) {
      const sf = n.getSourceFile();
      const rel = relOf(sf);
      const nm =
        (n.name && ts.isIdentifier(n.name) && n.name.text) ||
        (ts.isVariableDeclaration(n.parent) && ts.isIdentifier(n.parent.name) && n.parent.name.text) ||
        "";
      const fn = findNodeAt(rel, lineAt(sf, n.getStart()), nm || null);
      if (fn) return { node: fn, name: nm || containerName || "provider" };
    }
    if (ts.isClassDeclaration(n) && n.name) {
      const sf = n.getSourceFile();
      const rel = relOf(sf);
      const cls = (classesByFile.get(rel) || []).find((c) => key(c.label) === key(n.name.text));
      if (cls) return { node: cls.id, name: n.name.text };
    }
    n = n.parent;
  }
  const sf = decl.getSourceFile();
  const rel = relOf(sf);
  const fileNode = fileNodeByPath.get(rel);
  return fileNode ? { node: fileNode, name: path.basename(rel, path.extname(rel)) } : null;
}

function findDeclaringTarget(decl) {
  const sf = decl.getSourceFile();
  const rel = relOf(sf);
  if (ts.isMethodDeclaration(decl) || ts.isMethodSignature(decl) || ts.isGetAccessor(decl) || ts.isSetAccessor(decl)) {
    const owner = decl.parent;
    if (!owner || !owner.name) return null;
    const classes = (classesByFile.get(rel) || []).filter((n) => key(n.label) === key(owner.name.getText()));
    if (classes.length === 1) {
      const name = decl.name && ts.isIdentifier(decl.name) ? decl.name.text : null;
      if (name) return methodsByClass.get(classes[0].id)?.get(key(name)) || null;
    }
    return null;
  }
  if (ts.isPropertyAssignment(decl) || ts.isPropertyDeclaration(decl) || ts.isShorthandPropertyAssignment(decl)) {
    const value = ts.isShorthandPropertyAssignment(decl) ? null : decl.initializer;
    const methodLike =
      value && (ts.isArrowFunction(value) || ts.isFunctionExpression(value) || ts.isMethodDeclaration(value));
    if (!methodLike) return null;
    let objLiteral = null;
    if (ts.isPropertyAssignment(decl) && decl.parent && ts.isObjectLiteralExpression(decl.parent)) objLiteral = decl.parent;
    const provider = findProvider(decl, objLiteral);
    if (!provider) return null;
    return { propDecl: decl, provider, rel, sf };
  }
  if (ts.isFunctionDeclaration(decl) && decl.name) {
    return findNodeAt(rel, lineAt(sf, decl.getStart()), decl.name.text);
  }
  if (ts.isVariableDeclaration(decl) && ts.isIdentifier(decl.name)) {
    return findNodeAt(rel, lineAt(sf, decl.getStart()), decl.name.text);
  }
  return null;
}

const configPath = path.join(repo, "riveredge-frontend", "tsconfig.json");
const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
const parsed = ts.parseJsonConfigFileContent(configFile.config, ts.sys, path.dirname(configPath));
const rootNames = parsed.fileNames.filter((f) => /\.(ts|tsx)$/.test(f) && !f.endsWith(".d.ts"));
const program = ts.createProgram({
  rootNames,
  options: { ...parsed.options, noEmit: true, skipLibCheck: true, allowJs: false },
});
const checker = program.getTypeChecker();

function enclosingCallerNode(call, sf, rel) {
  let n = call.parent;
  while (
    n &&
    !(ts.isFunctionDeclaration(n) || ts.isMethodDeclaration(n) || ts.isArrowFunction(n) || ts.isFunctionExpression(n))
  ) {
    n = n.parent;
  }
  if (!n) return null;
  let line = lineAt(sf, n.getStart());
  let name = n.name && ts.isIdentifier(n.name) ? n.name.text : null;
  if (ts.isVariableDeclaration(n.parent) && ts.isIdentifier(n.parent.name)) {
    line = lineAt(sf, n.parent.getStart());
    name = n.parent.name.text;
  }
  return findNodeAt(rel, line, name) || findNodeAt(rel, line, null);
}

const added = [];
const seen = new Set();
function addCallEdge(caller, target, rel, line) {
  if (!caller || !target || caller === target) return false;
  const pair = `${caller}\u0000${target}`;
  if (pairs.has(pair) || seen.has(pair)) return false;
  seen.add(pair);
  added.push({
    source: caller,
    target,
    relation: "calls",
    confidence: "EXTRACTED",
    confidence_score: 1.0,
    source_file: rel,
    source_location: `L${line}`,
    context: "ts_instance_call_postpass",
    weight: 1.0,
    _origin: "ast",
  });
  return true;
}
for (const sf of program.getSourceFiles()) {
  if (sf.isDeclarationFile) continue;
  if (!sf.fileName.startsWith(path.join(repo, "riveredge-frontend"))) continue;
  const rel = relOf(sf);
  const visit = (node) => {
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression)) {
      const prop = node.expression.name;
      const sym = checker.getSymbolAtLocation(prop);
      const decls = sym ? sym.getDeclarations() || [] : [];
      for (const decl of decls) {
        const dsf = decl.getSourceFile();
        if (dsf.isDeclarationFile) continue;
        const drel = relOf(dsf);
        if (drel.startsWith("..") || drel.includes("node_modules")) continue;
        const resolved = findDeclaringTarget(decl);
        if (!resolved) continue;
        const caller = enclosingCallerNode(node, sf, rel);
        if (!caller) continue;
        let target = resolved.propDecl
          ? ensureMethodNode(resolved.rel, resolved.sf, resolved.propDecl, resolved.provider.name, resolved.provider.node)
          : resolved;
        if (!target) continue;
        const line = lineAt(sf, node.getStart());
        if (addCallEdge(caller, target, rel, line)) break;
      }
    } else if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
      const sym = checker.getSymbolAtLocation(node.expression);
      const decls = sym ? sym.getDeclarations() || [] : [];
      for (const decl of decls) {
        const dsf = decl.getSourceFile();
        if (dsf.isDeclarationFile) continue;
        const drel = relOf(dsf);
        if (drel.startsWith("..") || drel.includes("node_modules")) continue;
        const resolved = findDeclaringTarget(decl);
        if (!resolved || resolved.propDecl) continue;
        const caller = enclosingCallerNode(node, sf, rel);
        if (addCallEdge(caller, resolved, rel, lineAt(sf, node.getStart()))) break;
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
}

console.log(`ts files checked: ${rootNames.length} | new ts member-call edges: ${added.length} | new method nodes: ${newNodeIds.size}`);
for (const e of added.slice(0, 15)) {
  console.log(
    `  ${nodes.get(e.source).label} -> ${nodes.get(e.target).label} | ${e.source_file}:${e.source_location}`,
  );
}
if (write) {
  graph.links.push(...added);
  fs.writeFileSync(graphPath, JSON.stringify(graph));
  console.log(`wrote ${graphPath}`);
} else {
  console.log("dry run (pass --write to update graph.json)");
}
