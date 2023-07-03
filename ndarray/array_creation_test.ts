import { asserts } from "../deps.ts";

import { T } from "./types.ts";
import { arrayEqual, fromArray } from "./array.ts";
import {
  arange,
  copy,
  fromAny,
  fromFunction,
  ones,
  zeros,
} from "./array_creation.ts";

Deno.test("tests", () => {
  asserts.assert(arrayEqual(zeros([1, 2]), fromArray([[0, 0]])));
  asserts.assert(arrayEqual(ones([1, 2]), fromArray([[1, 1]])));

  asserts.assert(
    arrayEqual(fromFunction((idx) => idx[1], [1, 2]), fromArray([[0, 1]])),
  );
});

Deno.test("arange", () => {
  asserts.assert(arrayEqual(arange(3, 6, 2), fromArray([3, 5])));
});

Deno.test("fromAny", () => {
  asserts.assert(
    arrayEqual(
      fromAny([2, 3]),
      fromArray([2, 3]),
    ),
  );

  asserts.assert(
    arrayEqual(
      fromAny([
        fromArray([1, 2]),
        fromArray([3, 4]),
      ]),
      fromArray([[
        1,
        2,
      ], [3, 4]]),
    ),
  );
});

Deno.test("copy", () => {
  const a = fromArray([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const b = copy(a);

  asserts.assert(arrayEqual(a, b));
});
