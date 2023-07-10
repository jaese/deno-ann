import { asserts } from "../deps.ts";

import { erf } from "./erf.ts";

export function sampleNormalDist(mu: number, sigma: number): number {
  const r = Math.random();
  return inverseNormalCDF(r, mu, sigma, 0.01);
}

export function inverseNormalCDF(
  p: number,
  mu: number,
  sigma: number,
  tolerance: number,
): number {
  if (mu !== 0 || sigma !== 1) {
    return mu + sigma * inverseNormalCDF(p, 0, 1, tolerance);
  }

  let lowZ = -10.0;
  let hiZ = 10.0;
  while (true) {
    const midZ = (lowZ + hiZ) / 2;
    const midP = normalCDF(midZ, 0, 1);
    if (hiZ - lowZ <= tolerance) {
      return midZ;
    }
    if (midP < p) {
      lowZ = midZ;
    } else {
      hiZ = midZ;
    }
  }
}

export function normalCDF(x: number, mu: number, sigma: number): number {
  return (1 + erf((x - mu) / Math.sqrt(2) / sigma)) / 2;
}

export function getRandomInt(max: number): number {
  return Math.floor(Math.random() * max);
}

export function randomChoice<T>(arr: T[]): T {
  return arr[getRandomInt(arr.length)];
}

export function getRandomIntProbas(probas: number[]): number {
  const probasAcc = [] as number[];
  for (let i = 0; i < probas.length; i++) {
    const prev = i === 0 ? 0 : probasAcc[i - 1];
    probasAcc.push(probas[i] + prev);
  }
  asserts.assertAlmostEquals(probasAcc[probasAcc.length - 1], 1);

  const r = Math.random();
  for (let i = 0; i < probasAcc.length; i++) {
    if (r < probasAcc[i]) {
      return i;
    }
  }
  throw new Error("Not supposed to reach here");
}
