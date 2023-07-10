import { asserts } from "../deps.ts";

export {
  getRandomInt,
  getRandomIntProbas,
  inverseNormalCDF,
  normalCDF,
  randomChoice,
  sampleNormalDist,
} from "./random.ts";

const rtol = 1e-2;
const atol = 1e-2;

export function isclose(a: number, b: number): boolean {
  return Math.abs(a - b) <= (atol + rtol * Math.abs(b));
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
