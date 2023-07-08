import { asserts } from "../deps.ts";

import * as numerical from "../numerical/mod.ts";

import { arrayEqual, make, sizeOfShape } from "./array.ts";
import { T } from "./types.ts";
import { apply, elementwiseOp, reduceAxis, reduceAxis2 } from "./op.ts";

export function scale(a: T, b: number): T {
  return apply(a, (x) => x * b);
}

export function neg(a: T): T {
  return apply(a, (x) => -x);
}
export function sin(a: T): T {
  return apply(a, (x) => Math.sin(x));
}
export function cos(a: T): T {
  return apply(a, (x) => Math.cos(x));
}
export function isnan(a: T): T {
  return apply(a, (x) => isNaN(x) ? 1 : 0);
}

export function exp(a: T): T {
  return apply(a, Math.exp);
}
export function log(a: T): T {
  return apply(a, Math.log);
}

export function sumAll(a: T): number {
  const n = sizeOfShape(a.shape());
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += a.buffer()[i];
  }
  return s;
}

export function meanAll(a: T): number {
  const n = sizeOfShape(a.shape());
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += a.buffer()[i];
  }
  return s / n;
}

export function all(a: T): boolean {
  for (let i = 0; i < sizeOfShape(a.shape()); i++) {
    if (a.buffer()[i] === 0) {
      return false;
    }
  }
  return true;
}

export function any(a: T): boolean {
  for (let i = 0; i < sizeOfShape(a.shape()); i++) {
    if (a.buffer()[i] !== 0) {
      return true;
    }
  }
  return false;
}

export function max(a: T, axis: number): T {
  return reduceAxis(
    a,
    axis,
    (acc, x) => Math.max(acc, x),
    -Infinity,
  );
}

export function min(a: T, axis: number): T {
  return reduceAxis(
    a,
    axis,
    (acc, x) => Math.min(acc, x),
    Infinity,
  );
}

export function sum(a: T, axis: number): T {
  return reduceAxis(
    a,
    axis,
    (acc, x) => acc + x,
    0,
  );
}

export function mean(a: T, axis: number): T {
  const n = a.shape()[axis];
  return reduceAxis(
    a,
    axis,
    (acc, x) => acc + x / n,
    0,
  );
}

export function prod(a: T, axis: number): T {
  return reduceAxis(
    a,
    axis,
    (acc, x) => acc * x,
    1,
  );
}

export function argmax(a: T, axis: number): T {
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
}

export function argmin(a: T, axis: number): T {
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
}

// binary ops

export function add(a: T, b: T): T {
  return elementwiseOp(a, b, (x, y) => x + y);
}
export function sub(a: T, b: T): T {
  return elementwiseOp(a, b, (x, y) => x - y);
}
export function mul(a: T, b: T): T {
  return elementwiseOp(a, b, (x, y) => x * y);
}
export function div(a: T, b: T): T {
  return elementwiseOp(a, b, (x, y) => x / y);
}
export function power(a: T, b: T): T {
  return elementwiseOp(a, b, (x, y) => x ** y);
}
export function equal(a: T, b: T): T {
  return elementwiseOp(a, b, (x, y) => x === y ? 1 : 0);
}
export function isclose(a: T, b: T): T {
  return elementwiseOp(a, b, (x, y) => numerical.isclose(x, y) ? 1 : 0);
}

export function convolveValid(a: T, v: T): T {
  asserts.assertEquals(a.ndim(), 1);
  asserts.assertEquals(v.ndim(), 1);

  const m = a.shape()[0];
  const aBuffer = a.buffer();
  const n = v.shape()[0];
  const vBuffer = v.buffer();

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
}

export function assertArrayClose(
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

  message = `Array values are not close: ${
    JSON.stringify(Array.from(actual.buffer()))
  }, ${JSON.stringify(Array.from(expected.buffer()))}${msgSuffix}`;
  asserts.assert(
    all(isclose(actual, expected)),
    message,
  );
}
