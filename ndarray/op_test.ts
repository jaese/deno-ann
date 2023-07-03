import { asserts } from "../deps.ts";

import { T } from "./types.ts";
import { arrayEqual } from "./array.ts";
import { fromAny } from "./array_creation.ts";
import { apply, applyWithArrayResult, map } from "./op.ts";
import { mul } from "./math.ts";

Deno.test("map", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  const result = map((x: T): T => mul(x, fromAny([2])), a);

  asserts.assert(
    arrayEqual(
      result,
      fromAny([
        [2, 4, 6],
        [8, 10, 12],
      ]),
    ),
  );
});

Deno.test("apply", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  asserts.assert(arrayEqual(
    apply(a, (x) => x ** 2),
    fromAny([
      [1, 4, 9],
      [16, 25, 36],
    ]),
  ));
});

Deno.test("applyWithArrayResult", () => {
  asserts.assert(
    arrayEqual(
      applyWithArrayResult(
        fromAny([
          [1, 2],
          [3, 4],
        ]),
        3,
        (x) => [x, x * 2, x * 3],
      ),
      fromAny([
        [
          [1, 2, 3],
          [2, 4, 6],
        ],
        [
          [3, 6, 9],
          [4, 8, 12],
        ],
      ]),
    ),
  );
});
