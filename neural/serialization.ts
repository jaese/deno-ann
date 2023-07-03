import * as trees from "../trees/mod.ts";

import { Model } from "./types.ts";

export async function saveModelJSON(
  filepath: string,
  model: Model,
): Promise<void> {
  const ps = model.params()!;
  const bs = trees.jsonEncode(ps);
  await Deno.writeTextFile(filepath, bs);
}

export async function loadModelJSON(
  model: Model,
  filepath: string,
): Promise<void> {
  const bs = await Deno.readTextFile(filepath);
  const ps = trees.jsonDecode(bs);
  trees.set(model.params()!, ps);
}
