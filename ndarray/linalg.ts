import { asserts } from "../deps.ts";

import { Index, Shape, T } from "./types.ts";
import { make, sizeOfShape } from "./array.ts";

export const matmul = (a: T, b: T): T => {
  asserts.assertEquals(a.ndim(), 2);
  asserts.assertEquals(b.ndim(), 2);
  asserts.assertEquals(a.shape()[1], b.shape()[0]);

  const m = a.shape()[1];
  const newShape = [a.shape()[0], b.shape()[1]];

  const bufferA = a.buffer();
  const bufferB = b.buffer();

  const outBuffer = new Float32Array(sizeOfShape(newShape));

  for (let i = 0; i < a.shape()[0]; i++) {
    for (let k = 0; k < b.shape()[1]; k++) {
      let s = 0;
      for (let j = 0; j < m; j++) {
        s += bufferA[i * m + j] * bufferB[j * b.shape()[1] + k];
      }
      outBuffer[i * b.shape()[1] + k] = s;
    }
  }

  return make(newShape, outBuffer);
};
