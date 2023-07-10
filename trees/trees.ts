import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";

import { assertIsLeaf, assertIsMap, isLeaf, Path, T } from "./types.ts";

export function map(func: (xs: nd.T[]) => nd.T, trees: T[]): T {
  const firstTree = trees[0];
  if (isLeaf(firstTree)) {
    return func(trees as nd.T[]);
  } else if (firstTree instanceof Map) {
    const result = [] as [string, T][];
    for (const key of firstTree.keys()) {
      result.push(
        [
          key,
          map(
            func,
            trees.map((tree: T) => {
              assertIsMap(tree);
              asserts.assertEquals(firstTree.keys(), tree.keys());

              const result = tree.get(key)!;
              asserts.assertExists(result);
              return result;
            }),
          ),
        ],
      );
    }
    return new Map(result);
  } else if (firstTree instanceof Array) {
    const n = firstTree.length;
    for (let i = 1; i < trees.length; i++) {
      asserts.assert(trees[i] instanceof Array);
      const t = trees[i] as Array<T>;
      asserts.assertEquals(t.length, n);
    }

    const result = [] as T[];
    for (let i = 0; i < n; i++) {
      const y = map(
        func,
        trees.map((tree) => (tree as T[])[i]),
      );
      result.push(y);
    }
    return result;
  } else {
    throw new Error(firstTree);
  }
}

export function forEach(fun: (xs: nd.T[]) => void, trees: T[]): void {
  const empty = nd.zeros([]);
  const f = (xs: nd.T[]): nd.T => {
    fun(xs);
    return empty;
  };
  map(f, trees);
  return;
}

export function set(dst: T, src: T): void {
  const f = (xs: nd.T[]): nd.T => {
    const [d, s] = xs;

    // nd.copy(d, s);
    d.set([], s);

    return nd.fromNumber(0); // not used
  };

  map(f, [dst, src]);
}

export function copy(xs: T): T {
  return map(([x]) => nd.copy(x), [xs]);
}

export function equalAll(t1: T, t2: T): boolean {
  return all(map(([a, b]) => nd.equal(a, b), [t1, t2]));
}

export function assertEqualAll(actual: T, expected: T, msg?: string): void {
  const actualPaths = Array.from(iteratePaths(actual));
  const expectedPaths = Array.from(iteratePaths(expected));

  const msgSuffix = msg ? `: ${msg}` : ".";
  let message = `Tree shapes are not equal${msgSuffix}`;
  asserts.assertEquals(actualPaths, expectedPaths, message);

  for (const p of actualPaths) {
    const a = getLeafByPath(actual, p);
    const e = getLeafByPath(expected, p);

    message = `Elements at path ${p} are not equal${msgSuffix}`;
    nd.assertArrayEqual(a, e, message);
  }
}

export function assertCloseAll(actual: T, expected: T, msg?: string): void {
  const actualPaths = Array.from(iteratePaths(actual));
  const expectedPaths = Array.from(iteratePaths(expected));

  const msgSuffix = msg ? `: ${msg}` : ".";
  let message = `Tree shapes are not equal${msgSuffix}`;
  asserts.assertEquals(actualPaths, expectedPaths, message);

  for (const p of actualPaths) {
    const a = getLeafByPath(actual, p);
    const e = getLeafByPath(expected, p);

    message = `Elements at path ${p} are not close${msgSuffix}`;
    nd.assertArrayClose(a, e, message);
  }
}

export function all(t: T): boolean {
  const ls = leaves(t);
  for (const l of ls) {
    if (!nd.all(l)) {
      return false;
    }
  }
  return true;
}

export function leaves(t: T): nd.T[] {
  const result = [] as nd.T[];
  const walk = (x: T): void => {
    if (isLeaf(x)) {
      result.push(x);
    } else if (x instanceof Array) {
      for (const child of x) {
        walk(child);
      }
    } else if (x instanceof Map) {
      for (const [_key, value] of x) {
        walk(value);
      }
    } else {
      throw new Error(x);
    }
  };

  walk(t);
  return result;
}

export enum NodeType {
  Leaf = 0,
  Array,
  Map,
}

export interface TreeDef {
  nodeType: NodeType;
  nodeMetadata: string[] | null;
  childTreeDefs: TreeDef[];
}

export function reduce<Acc>(
  fun: (acc: Acc, xs: nd.T[]) => Acc,
  ts: T[],
  initial: Acc,
): Acc {
  let acc = initial;
  const f = (xs: nd.T[]): void => {
    acc = fun(acc, xs);
  };
  forEach(f, ts);
  return acc;
}

export function flatten(x: T): [nd.T[], TreeDef] {
  if (isLeaf(x)) {
    return [[x], {
      nodeType: NodeType.Leaf,
      nodeMetadata: null,
      childTreeDefs: [],
    }];
  } else if (x instanceof Array) {
    const flattened = [] as nd.T[];
    const childTrees = [] as TreeDef[];

    for (const child of x) {
      const [f, ct] = flatten(child);
      flattened.push(...f);
      childTrees.push(ct);
    }

    return [flattened, {
      nodeType: NodeType.Array,
      nodeMetadata: null,
      childTreeDefs: childTrees,
    }];
  } else if (x instanceof Map) {
    const nodeMetadata = [] as string[];
    const flattened = [] as nd.T[];
    const childTrees = [] as TreeDef[];

    const keys = Array.from(x.keys());
    keys.sort();

    for (const key of keys) {
      const value = x.get(key)!;

      nodeMetadata.push(key);
      const [f, ct] = flatten(value);
      flattened.push(...f);
      childTrees.push(ct);
    }

    return [flattened, {
      nodeType: NodeType.Map,
      nodeMetadata: nodeMetadata,
      childTreeDefs: childTrees,
    }];
  } else {
    throw new Error(x);
  }
}

export function unflatten(treedef: TreeDef, xs: nd.T[]): T {
  return unflattenInner(treedef, xs[Symbol.iterator]());
}

function unflattenInner(treedef: TreeDef, iter: Iterator<nd.T>): T {
  if (treedef.nodeType === NodeType.Leaf) {
    return iter.next().value;
  } else if (treedef.nodeType === NodeType.Array) {
    const result = [] as T[];
    for (const td of treedef.childTreeDefs) {
      result.push(unflattenInner(td, iter));
    }
    return result;
  } else if (treedef.nodeType === NodeType.Map) {
    asserts.assertEquals(
      treedef.nodeMetadata!.length,
      treedef.childTreeDefs.length,
    );

    const keys = treedef.nodeMetadata!;

    const result = new Map<string, T>();
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const value = unflattenInner(treedef.childTreeDefs[i], iter);
      result.set(key, value);
    }
    return result;
  } else {
    throw new Error(treedef.nodeType);
  }
}

export function iteratePaths(x: T): Iterable<Path> {
  const iterNode = function* (p: Path, rest: T): Iterable<Path> {
    if (isLeaf(rest)) {
      yield p;
    } else if (rest instanceof Array) {
      for (let i = 0; i < rest.length; i++) {
        for (const restPath of iterNode([...p, i], rest[i])) {
          yield restPath;
        }
      }
    } else if (rest instanceof Map) {
      const keys = Array.from(rest.keys());
      keys.sort();
      for (const key of keys) {
        for (const restPath of iterNode([...p, key], rest.get(key)!)) {
          yield restPath;
        }
      }
    } else {
      throw new Error(rest);
    }
  };

  return iterNode([], x);
}

export function getLeafByPath(x: T, p: Path): nd.T {
  if (p.length === 0) {
    assertIsLeaf(x);
    return x;
  }

  if (isLeaf(x)) {
    throw new Error(JSON.stringify(p));
  } else if (x instanceof Map) {
    const key = p[0];
    asserts.assert((typeof key) === "string");
    return getLeafByPath(x.get(key)!, p.slice(1));
  } else if (x instanceof Array) {
    const idx = p[0];
    asserts.assert(Number.isInteger(idx));
    return getLeafByPath(x[idx as number], p.slice(1));
  } else {
    throw new Error(x);
  }
}

export function plurality(x: T): number {
  if (isLeaf(x)) {
    asserts.assert(x.ndim() >= 1);
    return x.shape()[0];
  } else if (x instanceof Map) {
    for (const elem of x.values()) {
      return plurality(elem);
    }
    throw new Error(`${x}`);
  } else if (x instanceof Array) {
    return plurality(x[0]);
  } else {
    throw new Error(x);
  }
}
