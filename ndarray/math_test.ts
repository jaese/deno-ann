import { asserts } from "../deps.ts";

import { arrayEqual } from "./array.ts";
import { fromAny } from "./array_creation.ts";
import {
  all,
  argmax,
  argmin,
  convolveValid,
  equal,
  exp,
  log,
  max,
  min,
  prod,
  scale,
  sub,
  sum,
  sumAll,
} from "./math.ts";

Deno.test("sub", () => {
  asserts.assert(arrayEqual(
    sub(fromAny(3), fromAny([[0, 1], [4, 5]])),
    fromAny([[3, 2], [-1, -2]]),
  ));
});

Deno.test("ops", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const b = fromAny([
    [10, 11, 12],
  ]);

  asserts.assertEquals(sumAll(a), 21);
  asserts.assert(arrayEqual(max(a, 0), fromAny([4, 5, 6])));
  asserts.assert(arrayEqual(max(a, 1), fromAny([3, 6])));

  asserts.assert(arrayEqual(min(a, 0), fromAny([1, 2, 3])));
  asserts.assert(arrayEqual(min(a, 1), fromAny([1, 4])));

  asserts.assert(arrayEqual(sum(a, 0), fromAny([5, 7, 9])));
  asserts.assert(arrayEqual(sum(a, 1), fromAny([6, 15])));

  asserts.assert(arrayEqual(prod(a, 0), fromAny([4, 10, 18])));
  asserts.assert(arrayEqual(prod(a, 1), fromAny([6, 120])));

  asserts.assert(arrayEqual(argmax(a, 0), fromAny([1, 1, 1])));
  asserts.assert(arrayEqual(argmax(a, 1), fromAny([2, 2])));

  asserts.assert(arrayEqual(argmin(a, 0), fromAny([0, 0, 0])));
  asserts.assert(arrayEqual(argmin(a, 1), fromAny([0, 0])));

  asserts.assert(
    arrayEqual(
      equal(
        fromAny([1, 2]),
        fromAny([1, 0]),
      ),
      fromAny([
        1,
        0,
      ]),
    ),
  );

  asserts.assert(arrayEqual(
    scale(a, 2),
    fromAny([
      [2, 4, 6],
      [8, 10, 12],
    ]),
  ));

  asserts.assert(
    all(fromAny([1, 2])),
  );
  asserts.assertFalse(
    all(fromAny([1, 0])),
  );

  const nums = fromAny([1, 2]);

  asserts.assert(
    arrayEqual(exp(nums), fromAny([2.718281828459045, 7.38905609893065])),
  );
  asserts.assert(
    arrayEqual(log(nums), fromAny([0, 0.693147180])),
  );
});

Deno.test("convolveValid", () => {
  let input = fromAny([1, 2, 3, 4, 5]);
  let param = fromAny([1, 1, 1]);

  let result = convolveValid(input, param);

  asserts.assert(arrayEqual(
    result,
    fromAny([6, 9, 12]),
  ));

  input = fromAny([1, 2, 3]);
  param = fromAny([0, 1, 0.5]);

  result = convolveValid(input, param);

  asserts.assert(
    arrayEqual(
      result,
      fromAny([2.5]),
    ),
    `result: ${result.buffer()}`,
  );

  input = fromAny([0, 1, 2]);
  param = fromAny([1, 2, 3, 4, 5]);

  result = convolveValid(input, param);

  asserts.assert(arrayEqual(
    result,
    fromAny([4, 7, 10]),
  ));

  input = fromAny([1, 2, 3, 4, 5]);
  param = fromAny([0, 1, 2]);

  result = convolveValid(input, param);

  asserts.assert(arrayEqual(
    result,
    fromAny([4, 7, 10]),
  ));
});
