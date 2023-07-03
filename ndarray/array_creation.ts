import * as arrays from "../arrays/mod.ts";

import { isT, make, sizeOfShape } from "./array.ts";
import { Index, Shape, T } from "./types.ts";
import { fromArray, fromNumber } from "./array.ts";
import { concatenate, expandDims } from "./array_manipulation.ts";

export const fromAny = (v: any): T => {
  if (typeof (v) === "number") {
    return fromNumber(v);
  } else if (isT(v)) {
    return v;
  } else if (v instanceof Array) {
    const xs = [];
    for (const x of v) {
      xs.push(expandDims(fromAny(x), 0));
    }
    const result = concatenate(xs, 0);
    return result;
  } else {
    throw new Error(v);
  }
};

export const copy = (a: T): T => {
  const buffer = a.buffer().slice();
  return make(a.shape(), buffer);
};

export const arange = (start: number, end: number, step: number): T => {
  return fromArray(arrays.range(start, end, step));
};

export const zeros = (shape: number[]): T => {
  return make(shape, new Float32Array(sizeOfShape(shape)));
};

export const ones = (shape: number[]): T => {
  const result = make(shape, new Float32Array(sizeOfShape(shape)));
  for (let i = 0; i < result.buffer().length; i++) {
    result.buffer()[i] = 1;
  }
  return result;
};

export const fromFunction = (
  func: (idx: number[]) => number,
  shape: number[],
): T => {
  const buffer = new Float32Array(sizeOfShape(shape));
  applyFunction([], shape, buffer, func);
  return make(shape, buffer);
};

const applyFunction = (
  curIdx: number[],
  shape: Shape,
  buffer: Float32Array,
  func: (idx: number[]) => number,
): void => {
  if (shape.length === 0) {
    buffer[0] = func(curIdx);
    return;
  } else if (shape.length === 1) {
    for (let i = 0; i < shape[0]; i++) {
      buffer[i] = func([...curIdx, i]);
    }
    return;
  }

  for (let i = 0; i < shape[0]; i++) {
    const stride = sizeOfShape(shape.slice(1));
    applyFunction(
      [...curIdx, i],
      shape.slice(1),
      buffer.subarray(i * stride, (i + 1) * stride),
      func,
    );
  }
};
