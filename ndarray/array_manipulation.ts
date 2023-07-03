import { asserts } from "../deps.ts";

import * as arrays from "../arrays/mod.ts";

import { Index, Shape, T } from "./types.ts";
import { make, size, sizeOfShape } from "./array.ts";
import { zeros } from "./array_creation.ts";
import { copyWithPermutation } from "./op.ts";

export const reshape = (a: T, newShape: Shape): T => {
  asserts.assertEquals(sizeOfShape(newShape), sizeOfShape(a.shape()));

  return make(newShape, a.buffer());
};

export const expandDims = (a: T, axis: number): T => {
  return reshape(a, [...a.shape().slice(0, axis), 1, ...a.shape().slice(axis)]);
};

export const squeeze = (a: T, axis: number): T => {
  asserts.assertEquals(a.shape()[axis], 1);
  return reshape(a, [
    ...a.shape().slice(0, axis),
    ...a.shape().slice(axis + 1),
  ]);
};

export function concatenate(lst: T[], axis: number): T {
  asserts.assert(lst.length >= 1);

  const dims = [] as number[];
  const shapesRest = [] as Shape[];
  for (let i = 0; i < lst.length; i++) {
    const shp = lst[i].shape();
    asserts.assert(axis < shp.length);

    const d = shp[axis];
    const shapeRest = [...shp.slice(0, axis), ...shp.slice(axis + 1)];

    if (i >= 1) {
      if (!arrays.arrayEqual(shapesRest[0], shapeRest)) {
        throw Error(`${shapeRest} is different from ${shapesRest[0]}`);
      }
    }

    dims.push(d);
    shapesRest.push(shapeRest);
  }

  const dimsBefore = shapesRest[0].slice(0, axis);
  const sizeBefore = sizeOfShape(dimsBefore);

  const resultDim = dims.reduce((acc, x) => acc + x, 0);

  const dimsAfter = shapesRest[0].slice(axis);
  const strideAfter = sizeOfShape(dimsAfter);

  const resultSize = sizeBefore * resultDim * strideAfter;
  const strideBefore = resultDim * strideAfter;

  // console.log(resultSize, dimsBefore, dimsAfter);

  const buffer = new Float32Array(resultSize);
  let offset = 0;
  for (let i = 0; i < lst.length; i++) {
    buffer.set(lst[i].buffer(), offset);
    offset += lst[i].buffer().length;
  }

  for (let i = 0; i < sizeBefore; i++) {
    let curIdx = i * strideBefore;
    for (let j = 0; j < lst.length; j++) {
      const d = dims[j];

      const data = lst[j].buffer().subarray(
        i * d * strideAfter,
        (i + 1) * d * strideAfter,
      );

      buffer.subarray(curIdx, curIdx + d * strideAfter).set(data);

      curIdx += d * strideAfter;
    }
    asserts.assertEquals(curIdx, (i + 1) * strideBefore);
  }

  const newShape = [...dimsBefore, resultDim, ...dimsAfter];
  return make(newShape, buffer);
}

export const repeat = (a: T, repeats: number): T => {
  const shp = a.shape();
  const bufferA = a.buffer();
  const sz = size(a);
  const buffer = new Float32Array(sz * repeats);
  for (let i = 0; i < repeats; i++) {
    buffer.set(bufferA, sz * i);
  }
  const newShape = [repeats * shp[0], ...shp.slice(1)];
  return make(newShape, buffer);
};

/** Returns the view of `a` with the first dimension sliced [start, end). */
export const slice = (a: T, start: number, end: number): T => {
  asserts.assert(a.ndim() >= 1);
  const firstDim = a.shape()[0];
  asserts.assert(start >= 0 && start <= firstDim);
  asserts.assert(end >= 0, `failed: ${end} >= 0`);
  asserts.assert(end <= firstDim, `failed: ${end} <= ${firstDim}`);

  const restDims = a.shape().slice(1);
  const stride = sizeOfShape(restDims);
  // console.log(stride);
  // console.log(end - start);
  // console.log(start * stride, end * stride);

  const bufferView = a.buffer().subarray(start * stride, end * stride);
  // console.log(a.buffer());
  // console.log(bufferView);
  const newShape = [end - start, ...restDims];
  // console.log(newShape);

  return make(newShape, bufferView);
};

export const transpose = (a: T, axes: number[]): T => {
  asserts.assertEquals(a.shape().length, axes.length);

  const newShape = [] as Shape;
  for (let i = 0; i < axes.length; i++) {
    const k = a.shape()[axes[i]];
    newShape.push(k);
  }

  const buffer = new Float32Array(sizeOfShape(newShape));
  copyWithPermutation(
    buffer,
    newShape,
    a,
    axes,
    new Array(a.shape().length).fill(0),
  );

  return make(newShape, buffer);
};

export function swapaxes(a: T, axis1: number, axis2: number): T {
  const axes = arrays.range(0, a.ndim(), 1);
  axes[axis1] = axis2;
  axes[axis2] = axis1;
  return transpose(a, axes);
}

export const flip1D = (a: T): T => {
  asserts.assertEquals(a.ndim(), 1);

  const n = a.shape()[0];
  const bufferA = a.buffer();

  const buffer = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    buffer[i] = bufferA[n - i - 1];
  }

  return make([n], buffer);
};

export const pad1D = (a: T, num: number): T => {
  asserts.assertEquals(a.ndim(), 1);

  const padding = zeros([num]);
  return concatenate([
    padding,
    a,
    padding,
  ], 0);
};
