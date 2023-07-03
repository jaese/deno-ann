import { asserts } from "../deps.ts";

import * as trees from "../trees/mod.ts";
import * as nd from "../ndarray/mod.ts";

import { randomNormal } from "./inits.ts";
import { Model, Operation } from "./types.ts";
import { resetTree } from "./layer.ts";
import { mse } from "./metrics.ts";

export function numericalGrad(
  fun: (x: nd.T) => number,
  x: nd.T,
  h: number,
): nd.T {
  const g = nd.zeros(x.shape());
  const y = fun(x);

  for (const idx of nd.ndindex(x.shape())) {
    const xh = nd.copy(x);
    xh.set(idx, nd.fromAny(xh.get(idx).item() + h));

    const d = (fun(xh) - y) / h;

    g.set(idx, nd.fromAny(d));
  }

  return g;
}

export function numericalGradTree(
  fun: (inputs: trees.T) => number,
  inputs: trees.T,
  h: number,
): trees.T {
  const g = trees.map(([x]) => nd.zeros(x.shape()), [inputs]);

  for (const p of trees.iteratePaths(inputs)) {
    const x = trees.getLeafByPath(inputs, p);

    const inputsCopy = trees.copy(inputs);

    const funLeaf = (x: nd.T): number => {
      trees.getLeafByPath(inputsCopy, p).set([], x);
      return fun(inputsCopy);
    };

    const gLeaf = numericalGrad(funLeaf, x, h);
    trees.getLeafByPath(g, p).set([], gLeaf);
  }
  return g;
}

export function testOperationGrad(l: Operation, inputs: trees.T): void {
  asserts.assert(stacksIsEmpty(l.stacks()));
  const outputs = l.forward(inputs, true);
  asserts.assert(!stacksIsEmpty(l.stacks()));

  const g = trees.map(([x]) => randomNormal(x.shape(), 0, 1), [outputs]);
  const gradInputs = l.backward(g);
  asserts.assert(stacksIsEmpty(l.stacks()));

  const h = 0.001;

  const funInputs = (inputs: trees.T): number => {
    const outputs = l.forward(inputs, false);
    return trees.reduce<number>(
      (acc, [x, g]) => acc + nd.sumAll(nd.mul(x, g)),
      [
        outputs,
        g,
      ],
      0,
    );
  };

  const expectedGradInputs = numericalGradTree(funInputs, inputs, h);

  const diffMSE = trees.reduce((acc, [a, x]) => acc + mse(a, x), [
    gradInputs,
    expectedGradInputs,
  ], 0);
  console.log("grad inputs diff MSE:", diffMSE);

  trees.assertCloseAll(
    gradInputs,
    expectedGradInputs,
    "grad_inputs inconsistency",
  );
}

export function testModelParamGrads(l: Model, inputs: trees.T): void {
  const params = trees.copy(l.params());

  const outputs = l.forward(inputs, true);

  const g = trees.map(([x]) => randomNormal(x.shape(), 0, 1), [outputs]);
  resetTree(l.grads());
  l.backward(g);

  const gradParams = l.grads();

  const h = 0.001;

  const funParams = (params: trees.T): number => {
    trees.set(l.params(), params);
    const outputs = l.forward(inputs, false);
    return trees.reduce<number>(
      (acc, [x, g]) => acc + nd.sumAll(nd.mul(x, g)),
      [
        outputs,
        g,
      ],
      0,
    );
  };

  const expectedGradParams = numericalGradTree(funParams, params, h);

  const diffMSE = trees.reduce((acc, [a, x]) => acc + mse(a, x), [
    gradParams,
    expectedGradParams,
  ], 0);
  console.log("grad params diff MSE:", diffMSE);

  trees.assertCloseAll(
    gradParams,
    expectedGradParams,
    "grad_params inconsistency",
  );
}

export function testModelGrads(l: Model, inputs: trees.T): void {
  testOperationGrad(l, inputs);
  testModelParamGrads(l, inputs);
}

export function stacksIsEmpty(stacks: trees.T[][]): boolean {
  for (const stack of stacks) {
    if (stack.length !== 0) {
      return false;
    }
  }
  return true;
}
