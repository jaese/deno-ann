import * as nd from "../ndarray/mod.ts";

import { Dense, newConv1D } from "./layers.ts";
import { newSigmoid, newTanh } from "./activations.ts";
import { xavier } from "./inits.ts";
import { testModelGrads } from "./test_util.ts";

Deno.test("Dense", () => {
  const l = new Dense(3, 2, xavier, newTanh());
  const inputs = nd.fromAny([
    [-1, 2, 3],
    [4, 0, 6],
  ]);

  testModelGrads(l, inputs);

  testModelGrads(new Dense(3, 2, xavier, newSigmoid()), inputs);
  testModelGrads(new Dense(3, 3, xavier, null), inputs);
});

Deno.test("Conv1D", () => {
  const l = newConv1D(3, xavier);
  const x = nd.fromAny([1, 2, 3, 4, 5]);

  testModelGrads(l, x);
});
