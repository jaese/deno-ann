import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";

import { Loss } from "./types.ts";

export function softmax(arr: nd.T): nd.T {
  const largest = nd.expandDims(nd.max(arr, -1), arr.ndim());
  const exps = nd.exp(nd.sub(arr, largest));
  const sums = nd.sum(exps, -1);
  return nd.div(exps, nd.expandDims(sums, sums.ndim()));
}

export function computeAccuracy(yPred: nd.T, yTarget: nd.T): number {
  return nd.sumAll(nd.equal(yPred, yTarget)) / nd.size(yPred);
}

export function computeMulticlassAccuracy(
  logits: nd.T,
  labels: nd.T,
): number {
  asserts.assertEquals(logits.ndim(), labels.ndim() + 1);
  const preds = nd.argmax(logits, -1);
  return computeAccuracy(preds, labels);
}

export function sse(predicted: nd.T, target: nd.T): number {
  asserts.assertEquals(predicted.shape(), target.shape());

  const bufferPred = predicted.buffer();
  const bufferTarget = target.buffer();

  let s = 0;
  for (let i = 0; i < bufferPred.length; i++) {
    const e = bufferPred[i] - bufferTarget[i];
    s += e * e;
  }

  return s;
}

export function mse(predicted: nd.T, target: nd.T): number {
  return sse(predicted, target) / nd.size(predicted);
}

export const lossSSE = {
  loss(predicted: nd.T, target: nd.T): number {
    asserts.assertEquals(predicted.shape(), target.shape());

    const squaredErrors = nd.apply(
      nd.sub(predicted, target),
      (x) => (x * x) / 2,
    );

    return nd.sumAll(squaredErrors);
  },
  gradient(predicted: nd.T, target: nd.T): nd.T {
    asserts.assertEquals(predicted.ndim(), target.ndim());

    return nd.sub(predicted, target);
  },
} as Loss;

export const lossSoftmaxCrossEntropy = {
  loss(predicted: nd.T, target: nd.T): number {
    asserts.assertEquals(predicted.ndim(), target.ndim() + 1);
    const probas = softmax(predicted);

    const targetOneHot = oneHotEncode(
      target,
      probas.shape()[probas.ndim() - 1],
    );

    const likelihoods = nd.mul(
      nd.log(nd.add(probas, nd.fromNumber(1e-30))),
      targetOneHot,
    );
    asserts.assertEquals(likelihoods.shape(), probas.shape());

    return -nd.sumAll(likelihoods);
  },
  gradient(predicted: nd.T, target: nd.T): nd.T {
    const probas = softmax(predicted);
    const targetOneHot = oneHotEncode(
      target,
      probas.shape()[probas.ndim() - 1],
    );
    return nd.sub(probas, targetOneHot);
  },
} as Loss;

export function oneHotEncode(x: nd.T, numLabels: number): nd.T {
  const f = (x: number): number[] => {
    asserts.assert(Number.isInteger(x));
    asserts.assert(x < numLabels);
    const result = new Array(numLabels);
    for (let i = 0; i < numLabels; i++) {
      result[i] = i === x ? 1 : 0;
    }
    return result;
  };
  return nd.applyWithArrayResult(x, numLabels, f);
}
