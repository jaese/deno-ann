import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";

export type T = Map<string, T> | T[] | nd.T;

export const isLeaf = (tree: T): tree is nd.T => nd.isT(tree);

export function assertIsLeaf(tree: T): asserts tree is nd.T {
  asserts.assert(isLeaf(tree));
}

export function assertIsNotLeaf(tree: T): asserts tree is Map<string, T> | T[] {
  asserts.assert(!isLeaf(tree));
}

export function assertIsMap(tree: T): asserts tree is Map<string, T> {
  asserts.assert(tree instanceof Map);
}

export function assertIsArray(tree: T): asserts tree is T[] {
  asserts.assert(tree instanceof Array);
}

export type PathElement = string | number;
export type Path = PathElement[];
