import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";

import {
  newRelu,
  newSigmoid,
  newTanh,
  relu,
  sigmoid,
  tanh,
} from "./activations.ts";
import { testOperationGrad } from "./test_util.ts";

const testPoints = [
  -1,
  0,
  1,
  2,
];

Deno.test("test sigmoid", () => {
  const expected = [
    0.26894143,
    0.5,
    0.7310586,
    0.880797,
  ];

  testNumerical(sigmoid, testPoints, expected);
});

Deno.test("Sigmoid", () => {
  const l = newSigmoid();
  const input = nd.fromAny([[-1, 0, 2], [2, 1, 1]]);

  testOperationGrad(l, input);
});

const testNumerical = (
  func: (x: number) => number,
  testPoints: number[],
  expected: number[],
): void => {
  for (let i = 0; i < testPoints.length; i++) {
    asserts.assertAlmostEquals(func(testPoints[i]), expected[i]);
  }
};

Deno.test(
  "test tanh",
  () => {
    testNumerical(
      tanh,
      testPoints,
      [
        -0.7615941559557649,
        0,
        0.7615941559557649,
        0.9640275800758169,
      ],
    );
  },
);

Deno.test("Tanh", () => {
  const l = newTanh();
  const input = nd.fromAny([[-1, 0, 2], [-2, 1, 1]]);

  testOperationGrad(l, input);
});

Deno.test(
  "test relu",
  () => {
    testNumerical(
      relu,
      testPoints,
      [
        0,
        0,
        1,
        2,
      ],
    );
  },
);

Deno.test("Relu", () => {
  const l = newRelu();
  const input = nd.fromAny([[1, 0.1, 2], [2, 1, 1]]);

  testOperationGrad(l, input);
});
