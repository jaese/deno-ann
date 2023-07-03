import { asserts } from "../deps.ts";

export type { Index, Shape, T } from "./types.ts";
export {
  arrayEqual,
  assertArrayEqual,
  fromArray,
  fromNumber,
  isT,
  make,
  size,
  sizeOfShape,
  toArray,
} from "./array.ts";
export { copy, fromAny, fromFunction, ones, zeros } from "./array_creation.ts";
export {
  concatenate,
  expandDims,
  flip1D,
  pad1D,
  repeat,
  reshape,
  slice,
  squeeze,
  swapaxes,
  transpose,
} from "./array_manipulation.ts";
export {
  apply,
  applyWithArrayResult,
  combine,
  copyWithPermutation,
  elementwiseOp,
  reduceAxis,
  reduceAxis2,
} from "./op.ts";
export {
  add,
  all,
  any,
  argmax,
  argmin,
  assertArrayClose,
  convolveValid,
  cos,
  div,
  equal,
  exp,
  isclose,
  isnan,
  log,
  max,
  mean,
  meanAll,
  min,
  mul,
  neg,
  power,
  prod,
  scale,
  sin,
  sub,
  sum,
  sumAll,
} from "./math.ts";
export { matmul } from "./linalg.ts";
export { ndindex } from "./iterating.ts";
