import * as ndarray from "../ndarray/mod.ts";
import * as numerical from "../numerical/mod.ts";

export function randomUniform(shape: number[]): ndarray.T {
  return ndarray.fromFunction((_) => Math.random(), shape);
}

export function randomUniformBothWays(shape: number[]): ndarray.T {
  return ndarray.fromFunction((_) => Math.random() * 2 - 1, shape);
}

export function randomNormal(
  shape: number[],
  mu: number,
  sigma: number,
): ndarray.T {
  return ndarray.fromFunction(
    (_) => numerical.sampleNormalDist(mu, sigma),
    shape,
  );
}

export function xavier(shape: ndarray.Shape): ndarray.T {
  const variance = shape.length / shape.reduce((acc, x) => acc + x, 0);
  return randomNormal(shape, 0, variance);
}
