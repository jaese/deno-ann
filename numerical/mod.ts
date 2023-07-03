import { asserts } from "../deps.ts";

export { sampleNormalDist } from "./random.ts";

const rtol = 1e-2;
const atol = 1e-2;

export function isclose(a: number, b: number): boolean {
  return Math.abs(a - b) <= (atol + rtol * Math.abs(b));
}

export function arrayEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

export function mod(a: number, d: number): number {
  asserts.assert(d > 0);

  if (d === 1) {
    return 0;
  }

  if (a < 0) {
    return (a % d) + d;
  } else {
    return a % d;
  }
}
