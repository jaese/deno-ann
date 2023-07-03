import { asserts } from "../deps.ts";

import {
  arrayEqual,
  fromArray,
  fromNumber,
  size,
  sizeOfShape,
  toArray,
} from "./array.ts";

Deno.test("test stuffs", () => {
  asserts.assertEquals(sizeOfShape([1, 2, 5]), 10);

  asserts.assertEquals(
    arrayEqual(fromNumber(3), fromNumber(3)),
    true,
  );
  asserts.assertEquals(
    arrayEqual(
      fromArray([[1, 2, 3], [4, 5, 6]]),
      fromArray([[1, 2, 3], [4, 5, 6]]),
    ),
    true,
  );

  const a = fromArray([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  asserts.assertEquals(toArray(a), [[1, 2, 3], [4, 5, 6]]);

  asserts.assertEquals(toArray(a.get([1])), [4, 5, 6]);

  asserts.assertEquals(toArray(a.get([1, 2])), 6);
  asserts.assertEquals(a.get([1, 2]).item(), 6);

  asserts.assertEquals(size(a), 6);
});

Deno.test("set", () => {
  const a = fromArray([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  a.set([1, 2], fromNumber(7));
  asserts.assertEquals(a.get([1, 2]).item(), 7);

  a.set([1], fromArray([7, 8, 9]));
  asserts.assert(arrayEqual(a.get([1]), fromArray([7, 8, 9])));

  a.set([], fromArray([[1, 2, 3], [4, 5, 6]]));

  asserts.assert(arrayEqual(
    a,
    fromArray([[1, 2, 3], [4, 5, 6]]),
  ));

  a.set([], fromNumber(0));
  asserts.assert(arrayEqual(
    a,
    fromArray([[0, 0, 0], [0, 0, 0]]),
  ));
});
