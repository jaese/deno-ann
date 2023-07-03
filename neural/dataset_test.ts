import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";
import * as trees from "../trees/mod.ts";

import { dataLoaderFromDataset, TreeDataset } from "./dataset.ts";

Deno.test("TreeDataset", () => {
  const data = new Map<string, trees.T>([
    ["x", [nd.fromAny([[1, 2], [3, 4]]), nd.fromAny([5, 6])]],
    ["y", nd.fromAny([7, 8])],
  ]);

  const ds = new TreeDataset(data);

  asserts.assertEquals(ds.len(), 2);
  asserts.assert(trees.equalAll(
    ds.get(0),
    new Map<string, trees.T>([
      ["x", [nd.fromAny([1, 2]), nd.fromAny(5)]],
      ["y", nd.fromAny(7)],
    ]),
  ));
  asserts.assert(trees.equalAll(
    ds.get(1),
    new Map<string, trees.T>([
      ["x", [nd.fromAny([3, 4]), nd.fromAny(6)]],
      ["y", nd.fromAny(8)],
    ]),
  ));
});

Deno.test("dataLoaderFromDataset", () => {
  const data = new Map<string, trees.T>([
    ["x", [nd.fromAny([[1, 2], [3, 4]]), nd.fromAny([5, 6])]],
    ["y", nd.fromAny([7, 8])],
  ]);
  const ds = new TreeDataset(data);

  const dl = dataLoaderFromDataset(ds);
  const xs = Array.from(dl);

  asserts.assertEquals(xs.length, 2);
  asserts.assert(trees.equalAll(
    xs[0],
    new Map<string, trees.T>([
      ["x", [nd.fromAny([1, 2]), nd.fromAny(5)]],
      ["y", nd.fromAny(7)],
    ]),
  ));
  asserts.assert(trees.equalAll(
    xs[1],
    new Map<string, trees.T>([
      ["x", [nd.fromAny([3, 4]), nd.fromAny(6)]],
      ["y", nd.fromAny(8)],
    ]),
  ));
});
