import * as nd from "../ndarray/mod.ts";
import * as trees from "../trees/mod.ts";

export function resetTree(t: trees.T): void {
  trees.forEach(([x]: nd.T[]): void => x.set([], nd.fromAny(0)), [t]);
}
