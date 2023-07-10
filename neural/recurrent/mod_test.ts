import * as nd from "../../ndarray/mod.ts";
import * as neural from "../../neural/mod.ts";

import { GRUCell, RNNCell, SimpleRNN } from "./mod.ts";

Deno.test("SimpleRNN", () => {
  const numSteps = 2;
  const batchSize = 3;
  const inputSize = 4;
  const hiddenSize = 5;

  const l = new SimpleRNN(
    inputSize,
    hiddenSize,
    new GRUCell(inputSize, hiddenSize),
  );
  const xs = nd.fromAny([
    [
      [1, 2, 3, 4],
      [2, 0, 1, 2],
      [1, 1, 1, 0],
    ],
    [
      [0, 1, -1, -1],
      [1, -1, 1, 0],
      [0, -1, 1, 3],
    ],
  ]);
  const state = nd.zeros([batchSize, hiddenSize]);
  const inputs = [xs, state];

  neural.testModelGrads(l, inputs);
});

Deno.test("RNNCell", () => {
  const batchSize = 2;
  const inputSize = 3;
  const hiddenSize = 2;

  const l = new RNNCell(inputSize, hiddenSize, neural.xavier);
  // (batch_size, hidden_size)
  const xs = nd.fromAny([
    [1, 2, 3],
    [2, 0, 1],
  ]);
  const state = nd.zeros([batchSize, hiddenSize]);

  neural.testModelGrads(l, [xs, state]);
});

Deno.test("GRUCell", () => {
  const batchSize = 2;
  const inputSize = 3;
  const hiddenSize = 4;

  const l = new GRUCell(inputSize, hiddenSize);
  // (batch_size, hidden_size)
  const xs = nd.fromAny([
    [1, 2, 3],
    [2, 0, 1],
  ]);
  const state = nd.zeros([batchSize, hiddenSize]);

  neural.testModelGrads(l, [xs, state]);
});
