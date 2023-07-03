import { asserts } from "../deps.ts";

import * as numerical from "../numerical/mod.ts";

import { arrayEqual, fromNumber, make, sizeOfShape } from "./array.ts";
import { Index, Shape, T } from "./types.ts";
import { apply, elementwiseOp, reduceAxis, reduceAxis2 } from "./op.ts";

export const scale = (a: T, b: number): T => {
  return apply(a, (x) => x * b);
};

export const neg = (a: T): T => {
  return apply(a, (x) => -x);
};
export const sin = (a: T): T => {
  return apply(a, (x) => Math.sin(x));
};
export const cos = (a: T): T => {
  return apply(a, (x) => Math.cos(x));
};
export const isnan = (a: T): T => apply(a, (x) => isNaN(x) ? 1 : 0);

export const exp = (a: T): T => apply(a, Math.exp);
export const log = (a: T): T => apply(a, Math.log);

export const sumAll = (a: T): number => {
  const n = sizeOfShape(a.shape());
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += a.buffer()[i];
  }
  return s;
};

export const meanAll = (a: T): number => {
  const n = sizeOfShape(a.shape());
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += a.buffer()[i];
  }
  return s / n;
};

export const all = (a: T): boolean => {
  for (let i = 0; i < sizeOfShape(a.shape()); i++) {
    if (a.buffer()[i] === 0) {
      return false;
    }
  }
  return true;
};
export const any = (a: T): boolean => {
  for (let i = 0; i < sizeOfShape(a.shape()); i++) {
    if (a.buffer()[i] !== 0) {
      return true;
    }
  }
  return false;
};

export const max = (a: T, axis: number): T => {
  return reduceAxis(
    a,
    axis,
    (acc, x) => Math.max(acc, x),
    -Infinity,
  );
};

export const min = (a: T, axis: number): T => {
  return reduceAxis(
    a,
    axis,
    (acc, x) => Math.min(acc, x),
    Infinity,
  );
};

export const sum = (a: T, axis: number): T => {
  return reduceAxis(
    a,
    axis,
    (acc, x) => acc + x,
    0,
  );
};

export const mean = (a: T, axis: number): T => {
  const n = a.shape()[axis];
  return reduceAxis(
    a,
    axis,
    (acc, x) => acc + x / n,
    0,
  );
};

export const prod = (a: T, axis: number): T => {
  return reduceAxis(
    a,
    axis,
    (acc, x) => acc * x,
    1,
  );
};

export const argmax = (a: T, axis: number): T => {
  const f = (
    acc: [number, number],
    idx: number,
    x: number,
  ): [[number, number], number] => {
    if (x > acc[1]) {
      return [[idx, x], idx];
    } else {
      return [acc, acc[0]];
    }
  };
  return reduceAxis2(
    a,
    axis,
    f,
    [-1, -Infinity],
  );
};

export const argmin = (a: T, axis: number): T => {
  const f = (
    acc: [number, number],
    idx: number,
    x: number,
  ): [[number, number], number] => {
    if (x < acc[1]) {
      return [[idx, x], idx];
    } else {
      return [acc, acc[0]];
    }
  };
  return reduceAxis2(
    a,
    axis,
    f,
    [-1, Infinity],
  );
};

// binary mutate ops

// export const iadd = (dst: T, src: T) =>
//   elementwiseOpMutate(dst, src, (x, y) => x + y);

// binary ops

export const add = (a: T, b: T) => elementwiseOp(a, b, (x, y) => x + y);
export const sub = (a: T, b: T) => elementwiseOp(a, b, (x, y) => x - y);
export const mul = (a: T, b: T) => elementwiseOp(a, b, (x, y) => x * y);
export const div = (a: T, b: T) => elementwiseOp(a, b, (x, y) => x / y);
export const power = (a: T, b: T) => elementwiseOp(a, b, (x, y) => x ** y);
export const equal = (a: T, b: T) =>
  elementwiseOp(a, b, (x, y) => x === y ? 1 : 0);
export const isclose = (a: T, b: T): T =>
  elementwiseOp(a, b, (x, y) => numerical.isclose(x, y) ? 1 : 0);

export const convolveValid = (a: T, v: T): T => {
  asserts.assertEquals(a.ndim(), 1);
  asserts.assertEquals(v.ndim(), 1);
  // ...
  // asserts.assert(a.shape()[a.ndim() - 1] >= v.shape()[0]);

  const m = a.shape()[0];
  const aBuffer = a.buffer();
  const n = v.shape()[0];
  const vBuffer = v.buffer();

  // asserts.assert(m >= n);
  if (m >= n) {
    const outSize = m - n + 1;

    const buffer = new Float32Array(outSize);
    for (let i = 0; i < outSize; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) {
        const idxA = i + j;
        s += aBuffer[idxA] * vBuffer[n - j - 1];
      }
      buffer[i] = s;
    }

    return make([outSize], buffer);
  } else {
    return convolveValid(v, a);
  }
};

export const assertArrayClose = (
  actual: T,
  expected: T,
  msg?: string,
): void => {
  if (arrayEqual(actual, expected)) {
    return;
  }

  const msgSuffix = msg ? `: ${msg}` : ".";

  let message = `Shapes are not equal${msgSuffix}`;
  asserts.assertEquals(actual.shape(), expected.shape(), message);

  message = `Array values are not close: ${
    JSON.stringify(Array.from(actual.buffer()))
  }, ${JSON.stringify(Array.from(expected.buffer()))}${msgSuffix}`;
  asserts.assert(
    all(isclose(actual, expected)),
    message,
  );
};
