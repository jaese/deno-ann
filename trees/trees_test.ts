import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";

import { T } from "./types.ts";
import {
  copy,
  equalAll,
  flatten,
  getLeafByPath,
  iteratePaths,
  leaves,
  map,
  plurality,
  reduce,
  set,
  unflatten,
} from "./trees.ts";

Deno.test("Test equal", () => {
  const a = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);

  const b = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);

  const c = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(7), nd.fromAny(6)]],
  ]);

  asserts.assert(equalAll(a, b));
  asserts.assertFalse(equalAll(a, c));
});

Deno.test("leaves", () => {
  const a = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);

  const result = leaves(a);

  const expected = [
    nd.fromAny([2, 3]),
    nd.fromAny(4),
    nd.fromAny(5),
    nd.fromAny(6),
  ];
  asserts.assert(equalAll(result, expected));
});

Deno.test("Test map", () => {
  const tree = new Map<string, T>([
    ["weight", nd.fromArray([1, 2])],
    ["bias", nd.fromArray(7)],
    ["params", [
      nd.fromAny(3),
      nd.fromAny(4),
    ]],
  ]);

  const result = map(([x]) => nd.scale(x, 3), [tree]);

  asserts.assert(equalAll(
    result,
    new Map<string, T>([
      ["weight", nd.fromAny([3, 6])],
      ["bias", nd.fromAny(21)],
      ["params", [
        nd.fromAny(9),
        nd.fromAny(12),
      ]],
    ]),
  ));
});

Deno.test("Test set", () => {
  const dst = new Map([
    ["weight", nd.fromArray([1, 2, 3])],
    ["bias", nd.fromNumber(7)],
  ]);

  const src = new Map([
    ["weight", nd.fromArray([3, 2, 1])],
    ["bias", nd.fromNumber(3)],
  ]);

  set(dst, src);

  asserts.assert(equalAll(
    dst,
    src,
  ));
});

Deno.test("Test flatten and unflatten", () => {
  const a = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);

  const [flattened, treeDefs] = flatten(a);

  const unflattened = unflatten(treeDefs, flattened);

  asserts.assert(equalAll(a, unflattened));
});

Deno.test("iteratePaths", () => {
  const a = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);

  const result = Array.from(iteratePaths(a));

  asserts.assertEquals(
    result,
    [["W"], ["b"], ["params", 0], ["params", 1]],
  );
});

Deno.test("getLeafByPath", () => {
  const a = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);

  const result = getLeafByPath(a, ["params", 1]);

  asserts.assertEquals(result.item(), 6);
});

Deno.test("copy", () => {
  const a = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);
  const result = copy(a);

  asserts.assert(equalAll(
    result,
    a,
  ));
});

Deno.test("reduce", () => {
  const a = new Map<string, T>([
    ["W", nd.fromAny([2, 3])],
    ["b", nd.fromAny(4)],
    ["params", [nd.fromAny(5), nd.fromAny(6)]],
  ]);

  const result = reduce<number>(
    (acc: number, xs: nd.T[]): number => acc + nd.sumAll(xs[0]),
    [a],
    0,
  );

  asserts.assertEquals(result, 20);
});

Deno.test("plurality", () => {
  const a = new Map<string, T>([
    ["input1", nd.fromAny([2, 3])],
    ["input2", nd.fromAny([1, 4])],
    ["target", [nd.fromAny([0, 1]), nd.fromAny([2, 3])]],
  ]);

  asserts.assertEquals(plurality(a), 2);
  asserts.assertEquals(plurality(a.get("input1")!), 2);
  asserts.assertEquals(plurality(a.get("target")!), 2);
});
