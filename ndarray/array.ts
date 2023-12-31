import { asserts } from "../deps.ts";

import * as arrays from "../arrays/mod.ts";
import * as floats from "../floats/mod.ts";

import { Index, Shape, T } from "./types.ts";

export function size(a: T): number {
  return sizeOfShape(a.shape());
}

export function sizeOfShape(shape: number[]): number {
  return shape.reduce((acc, x) => acc * x, 1);
}

export function arrayEqual(a: T, b: T): boolean {
  if (!arrays.arrayEqual(a.shape(), b.shape())) {
    return false;
  }

  return floats.arrayEqual(a.buffer(), b.buffer());
}

export function shapeEqual(a: Shape, b: Shape): boolean {
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

export function isT(x: any): x is T {
  if ("@type" in x) {
    return x["@type"] === "ndarray";
  }
  return false;
}

export function assertArrayEqual(
  actual: T,
  expected: T,
  msg?: string,
): void {
  if (arrayEqual(actual, expected)) {
    return;
  }

  const msgSuffix = msg ? `: ${msg}` : ".";

  let message = `Shapes are not equal${msgSuffix}`;
  asserts.assertEquals(actual.shape(), expected.shape(), message);

  message = `Array values are not equal${msgSuffix}`;
  asserts.assertEquals(
    Array.from(actual.buffer()),
    Array.from(expected.buffer()),
    message,
  );
}

export function toArray(a: T): any {
  if (a.shape().length === 0) {
    return a.item();
  }
  const result = [];
  for (let i = 0; i < a.shape()[0]; i++) {
    result.push(toArray(a.get([i])));
  }
  return result;
}

export function make(shape: number[], buffer: Float32Array): T {
  asserts.assertEquals(sizeOfShape(shape), buffer.length);

  const a = {
    "@type": "ndarray",
    _shape: shape,
    _buffer: buffer,

    shape(): Shape {
      return this._shape;
    },
    ndim(): number {
      return this._shape.length;
    },
    buffer(): Float32Array {
      return this._buffer;
    },

    item(): number {
      if (this._shape.length !== 0) {
        const msg = `Single-item array expected; got ${
          JSON.stringify(this._shape)
        }.`;
        throw new Error(msg);
      }
      return this._buffer[0];
    },

    get(idx: number[]): T {
      // console.log("ENTER get", this, idx);
      const s = getSubarray(this._buffer, this._shape, idx);

      const t = make(this._shape.slice(idx.length), s);
      return t;
    },

    set(idx: number[], v: T): void {
      const s = this.get(idx);

      if (shapeEqual(v.shape(), s.shape())) {
        const dstBuffer = s.buffer();
        const srcBuffer = v.buffer();
        asserts.assertEquals(dstBuffer.length, srcBuffer.length);
        dstBuffer.set(srcBuffer);
        return;
      }

      copyWithBroadcast(s, v);
    },

    reshape(shape: number[]): void {
      asserts.assertEquals(sizeOfShape(shape), sizeOfShape(this._shape));

      this._shape = shape;
    },

    toString(): string {
      return "?";
    },
  };

  return a;
}

function getSubarray(
  arr: Float32Array,
  shape: number[],
  idx: number[],
): Float32Array {
  if (idx.length === 0) {
    return arr;
  }

  const i = idx[0];
  const stride = shape.slice(1).reduce((acc, x) => acc * x, 1);
  return getSubarray(
    arr.subarray(i * stride, (i + 1) * stride),
    shape.slice(1),
    idx.slice(1),
  );
}

function copyWithBroadcast(dst: T, src: T): void {
  [dst, src] = matchDimsForOp(dst, src);
  asserts.assertEquals(dst.shape().length, src.shape().length);

  if (dst.shape().length === 0) {
    dst.buffer()[0] = src.buffer()[0];
    return;
  }

  let srcStep = 1;
  if (dst.shape()[0] !== 1 && src.shape()[0] === 1) {
    // broadcast
    srcStep = 0;
  } else {
    asserts.assertEquals(dst.shape()[0], src.shape()[0]);
  }

  if (dst.shape().length === 1) {
    for (let i = 0; i < dst.shape()[0]; i++) {
      dst.buffer()[i] = src.buffer()[i * srcStep];
    }
    return;
  }

  for (let i = 0; i < dst.shape()[0]; i++) {
    copyWithBroadcast(
      dst.get([i]),
      src.get([i * srcStep]),
    );
  }
}

function matchDimsForOp(a: T, b: T): [T, T] {
  const targetNumDims = Math.max(a.shape().length, b.shape().length);
  return [extendDimForOp(a, targetNumDims), extendDimForOp(b, targetNumDims)];
}

function extendDimForOp(a: T, target: number): T {
  asserts.assert(a.shape().length <= target);

  const newShape = [...new Array(target - a.ndim()).fill(1), ...a.shape()];

  return make(newShape, a.buffer());
}

export function fromNumber(v: number): T {
  return make([], new Float32Array([v]));
}

export function fromArray(arr: any): T {
  const shape = [];

  let size = 1;
  let a = arr;
  while (Array.isArray(a)) {
    shape.push(a.length);
    size = size * a.length;
    a = a[0];
  }

  const buffer = new Float32Array(size);

  if (shape.length === 0) {
    buffer[0] = arr;
  } else {
    setArrayFromList(buffer, arr, shape);
  }

  return make(shape, buffer);
}

function setArrayFromList(
  buffer: Float32Array,
  arr: any,
  shape: Shape,
) {
  if (shape.length === 1) {
    for (let i = 0; i < arr.length; i++) {
      buffer[i] = arr[i];
    }
    return;
  }

  const subshape = shape.slice(1);
  const stride = subshape.reduce((acc, x) => acc * x, 1);
  for (let i = 0; i < arr.length; i++) {
    setArrayFromList(
      buffer.subarray(i * stride, (i + 1) * stride),
      arr[i],
      shape.slice(1),
    );
  }
}
