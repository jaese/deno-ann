import { asserts } from "../deps.ts";

import { arrayEqual } from "./array.ts";
import { fromAny } from "./array_creation.ts";
import { transpose } from "./array_manipulation.ts";
import { matmul } from "./linalg.ts";

Deno.test("matmul", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const b = fromAny([
    [10, 11, 12],
  ]);

  asserts.assert(
    arrayEqual(
      matmul(a, transpose(b, [1, 0])),
      fromAny([
        [68],
        [167],
      ]),
    ),
  );
});
