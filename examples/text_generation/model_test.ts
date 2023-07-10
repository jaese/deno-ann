import * as nd from "../../ndarray/mod.ts";
import * as neural from "../../neural/mod.ts";

import { Model } from "./model.ts";

Deno.test("Model", () => {
  const vocabSize = 4;
  const embedSize = 5;
  const numHiddens = 6;

  const model = new Model(vocabSize, embedSize, numHiddens);
  const x = nd.fromArray([
    [0, 1, 2],
    [2, 3, 1],
  ]);

  neural.testModelParamGrads(model, x);
});
