import { asserts } from "../deps.ts";

import * as numerical from "../numerical/mod.ts";

import { Index, Shape, T } from "./types.ts";
import { make, size, sizeOfShape } from "./array.ts";
import { zeros } from "./array_creation.ts";
import { concatenate, expandDims } from "./array_manipulation.ts";

export const reduceAxis2 = <Acc>(
  a: T,
  axis: number,
  func: (acc: Acc, idx: number, x: number) => [Acc, number],
  initialValue: Acc,
): T => {
  // console.log(axis, a.shape.length);
  const ax = numerical.mod(axis, a.shape().length);
  // console.log("ax", ax);
  const newShape = [...a.shape().slice(0, ax), ...a.shape().slice(ax + 1)];
  const buffer = new Float32Array(sizeOfShape(newShape));
  const dst = make(newShape, buffer);
  reduceAxis2Inner(
    dst,
    a,
    ax,
    func,
    initialValue,
  );
  return dst;
};

const reduceAxis2Inner = <Acc>(
  dst: T,
  a: T,
  axis: number,
  func: (acc: Acc, idx: number, x: number) => [Acc, number],
  initialValue: Acc,
) => {
  if (axis === 0) {
    const stride = sizeOfShape(a.shape().slice(1));
    asserts.assertEquals(dst.buffer().length, stride);
    for (let i = 0; i < stride; i++) {
      let acc = initialValue;
      let value = 0; // no value.
      for (let j = 0; j < a.shape()[0]; j++) {
        const idx = i + j * stride;
        [acc, value] = func(acc, j, a.buffer()[idx]);
      }
      dst.buffer()[i] = value;
    }
  } else {
    for (let i = 0; i < a.shape()[0]; i++) {
      reduceAxis2Inner(
        dst.get([i]),
        a.get([i]),
        axis - 1,
        func,
        initialValue,
      );
    }
  }
};

export const reduceAxis = (
  a: T,
  axis: number,
  func: (acc: number, x: number) => number,
  initialValue: number,
): T => {
  const ax = numerical.mod(axis, a.shape().length);
  const newShape = [...a.shape().slice(0, ax), ...a.shape().slice(ax + 1)];
  const buffer = new Float32Array(sizeOfShape(newShape));
  const dst = make(newShape, buffer);
  reduceAxisInner(
    dst,
    a,
    ax,
    func,
    initialValue,
  );
  return dst;
};

const reduceAxisInner = (
  dst: T,
  a: T,
  axis: number,
  func: (acc: number, x: number) => number,
  initialValue: number,
): void => {
  if (axis === 0) {
    const stride = sizeOfShape(a.shape().slice(1));
    asserts.assertEquals(dst.buffer().length, stride);
    for (let i = 0; i < stride; i++) {
      let acc = initialValue;
      for (let j = 0; j < a.shape()[0]; j++) {
        acc = func(acc, a.buffer()[i + j * stride]);
      }
      dst.buffer()[i] = acc;
    }
  } else {
    for (let i = 0; i < a.shape()[0]; i++) {
      reduceAxisInner(
        dst.get([i]),
        a.get([i]),
        axis - 1,
        func,
        initialValue,
      );
    }
  }
};

export const applyWithArrayResult = (
  a: T,
  dim: number,
  f: (x: number) => number[],
): T => {
  const buffer = new Float32Array(size(a) * dim);
  for (let i = 0; i < a.buffer().length; i++) {
    const resultArr = f(a.buffer()[i]);
    for (let j = 0; j < dim; j++) {
      buffer[i * dim + j] = resultArr[j];
    }
  }
  return make([...a.shape(), dim], buffer);
};

export const copyWithPermutation = (
  buffer: Float32Array,
  newShape: Shape,
  a: T,
  axes: number[],
  idx: Index,
): void => {
  if (axes.length === 0) {
    buffer[0] = a.get(idx).item();
    return;
  }

  const k = newShape[0];
  const stride = sizeOfShape(newShape.slice(1));
  for (let i = 0; i < k; i++) {
    idx[axes[0]] = i;
    copyWithPermutation(
      buffer.subarray(i * stride, (i + 1) * stride),
      newShape.slice(1),
      a,
      axes.slice(1),
      idx,
    );
  }
};

export const elementwiseOp = (
  a: T,
  b: T,
  func: (x: number, y: number) => number,
): T => {
  const [aExtended, bExtended] = matchDimsForOp(a, b);
  const resultShape = shapeFromOpWithBroadcast(
    aExtended.shape(),
    bExtended.shape(),
  );
  const result = zeros(resultShape);
  opWithBroadcast(
    result,
    aExtended,
    bExtended,
    func,
  );
  return result;
};

// TODO: reconcile with one in array.ts
const matchDimsForOp = (a: T, b: T): [T, T] => {
  const targetNumDims = Math.max(a.shape().length, b.shape().length);
  return [extendDimForOp(a, targetNumDims), extendDimForOp(b, targetNumDims)];
};

const extendDimForOp = (a: T, target: number): T => {
  asserts.assert(a.shape().length <= target);

  const newShape = [...new Array(target - a.ndim()).fill(1), ...a.shape()];

  return make(newShape, a.buffer());
};

const shapeFromOpWithBroadcast = (s1: Shape, s2: Shape): Shape => {
  asserts.assertEquals(s1.length, s2.length);

  const result = [] as number[];

  for (let i = 0; i < s1.length; i++) {
    if (s1[i] === 1 && s2[i] !== 1) {
      result.push(s2[i]);
    } else if (s1[i] !== 1 && s2[i] === 1) {
      result.push(s1[i]);
    } else {
      asserts.assertEquals(s1[i], s2[i]);
      result.push(s1[i]);
    }
  }

  return result;
};

const opWithBroadcast = (
  dst: T,
  a: T,
  b: T,
  func: (x: number, y: number) => number,
): void => {
  // console.log("copyWithBroadcast", dst, src);
  asserts.assertEquals(a.shape().length, b.shape().length);

  if (dst.shape().length === 0) {
    dst.buffer()[0] = func(a.buffer()[0], b.buffer()[0]);
    return;
  }

  let aStep = 1;
  let bStep = 1;
  if (a.shape()[0] !== 1 && b.shape()[0] === 1) {
    // broadcast
    bStep = 0;
  } else if (a.shape()[0] === 1 && b.shape()[0] !== 1) {
    // broadcast
    aStep = 0;
  } else {
    asserts.assertEquals(a.shape()[0], b.shape()[0]);
  }

  if (dst.shape().length === 1) {
    for (let i = 0; i < dst.shape()[0]; i++) {
      dst.buffer()[i] = func(a.buffer()[i * aStep], b.buffer()[i * bStep]);
    }
    return;
  }

  for (let i = 0; i < dst.shape()[0]; i++) {
    opWithBroadcast(
      dst.get([i]),
      a.get([i * aStep]),
      b.get([i * bStep]),
      func,
    );
  }
};

export const apply = (a: T, f: (x: number) => number): T => {
  const result = zeros(a.shape());
  for (let i = 0; i < size(a); i++) {
    result.buffer()[i] = f(a.buffer()[i]);
  }
  return result;
};

export const combine = (a: T, b: T, f: (x: number, y: number) => number): T => {
  asserts.assertEquals(a.shape(), b.shape());

  const result = zeros(a.shape());
  for (let i = 0; i < size(a); i++) {
    result.buffer()[i] = f(a.buffer()[i], b.buffer()[i]);
  }
  return result;
};

export const map = (fun: (x: T) => T, xs: T): T => {
  const result = [];
  for (let i = 0; i < xs.shape()[0]; i++) {
    const y = fun(xs.get([i]));
    result.push(expandDims(y, 0));
  }
  return concatenate(result, 0);
};
