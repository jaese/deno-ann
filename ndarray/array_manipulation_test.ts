import { asserts } from "../deps.ts";

import { arrayEqual } from "./array.ts";
import { fromAny } from "./array_creation.ts";
import {
  concatenate,
  expandDims,
  flip1D,
  pad1D,
  repeat,
  reshape,
  slice,
  squeeze,
  swapaxes,
  transpose,
} from "./array_manipulation.ts";

Deno.test("reshape", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  asserts.assert(
    arrayEqual(reshape(a, [3, 2]), fromAny([[1, 2], [3, 4], [5, 6]])),
  );
});
Deno.test("expandDims and squeeze", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  const b = expandDims(a, a.ndim());
  asserts.assertEquals(b.shape(), [2, 3, 1]);

  const c = squeeze(b, b.ndim() - 1);
  asserts.assertEquals(c.shape(), [2, 3]);
});

Deno.test("transpose", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  asserts.assert(
    arrayEqual(
      transpose(a, [1, 0]),
      fromAny([
        [1, 4],
        [2, 5],
        [3, 6],
      ]),
    ),
  );

  asserts.assert(
    arrayEqual(
      transpose(a, [0, 1]),
      a,
    ),
  );
});

Deno.test("Test slice", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
  ]);

  const result = slice(a, 1, 3);

  asserts.assert(
    arrayEqual(
      result,
      fromAny([
        [4, 5, 6],
        [7, 8, 9],
      ]),
    ),
  );
});

Deno.test("pad1D", () => {
  const a = fromAny([1, 2, 3]);
  const result = pad1D(a, 2);

  asserts.assert(arrayEqual(
    result,
    fromAny([0, 0, 1, 2, 3, 0, 0]),
  ));
});

Deno.test("flip1D", () => {
  const a = fromAny([1, 2, 3]);
  const result = flip1D(a);

  asserts.assert(arrayEqual(result, fromAny([3, 2, 1])));
});

Deno.test("swapaxes", () => {
  const a = fromAny([
    [
      [1, 2, 7],
      [4, 9, 6],
    ],
    [
      [0, 1, 2],
      [1, 2, 3],
    ],
  ]);
  const result = swapaxes(a, 0, 1);

  asserts.assert(arrayEqual(
    result,
    fromAny([
      [
        [1, 2, 7],
        [0, 1, 2],
      ],
      [
        [4, 9, 6],
        [1, 2, 3],
      ],
    ]),
  ));
});

Deno.test("repeat", () => {
  const a = fromAny([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  const result = repeat(a, 2);

  asserts.assert(arrayEqual(
    result,
    fromAny([
      [1, 2, 3],
      [4, 5, 6],
      [1, 2, 3],
      [4, 5, 6],
    ]),
  ));
});

Deno.test("concatenate", () => {
  {
    const a = fromAny([
      [
        [1, 2, 7],
        [4, 9, 6],
      ],
      [
        [0, 1, 2],
        [1, 2, 3],
      ],
    ]);

    const b = fromAny([
      [
        [10, 11, 12],
      ],
      [
        [-2, -1, 0],
      ],
    ]);

    asserts.assert(arrayEqual(
      concatenate([a, b], 1),
      fromAny([
        [
          [1, 2, 7],
          [4, 9, 6],
          [10, 11, 12],
        ],
        [
          [0, 1, 2],
          [1, 2, 3],
          [-2, -1, 0],
        ],
      ]),
    ));
  }

  {
    const a = fromAny([
      [
        [1, 2, 7],
        [4, 9, 6],
      ],
      [
        [0, 1, 2],
        [1, 2, 3],
      ],
    ]);

    const b = fromAny([
      [
        [10, 11, 12],
        [1, 2, 3],
      ],
      [
        [-2, -1, 0],
        [1, 2, 3],
      ],
    ]);

    asserts.assert(arrayEqual(
      concatenate([a, b], 0),
      fromAny([
        [
          [1, 2, 7],
          [4, 9, 6],
        ],
        [
          [0, 1, 2],
          [1, 2, 3],
        ],
        [
          [10, 11, 12],
          [1, 2, 3],
        ],
        [
          [-2, -1, 0],
          [1, 2, 3],
        ],
      ]),
    ));
  }

  {
    const a = fromAny([
      [
        [1, 2, 7],
        [4, 9, 6],
      ],
      [
        [0, 1, 2],
        [1, 2, 3],
      ],
    ]);

    const b = fromAny([
      [
        [10, 11, 12],
        [1, 2, 3],
      ],
      [
        [-2, -1, 0],
        [1, 2, 3],
      ],
    ]);

    asserts.assert(arrayEqual(
      concatenate([a, b], 2),
      fromAny([
        [
          [1, 2, 7, 10, 11, 12],
          [4, 9, 6, 1, 2, 3],
        ],
        [
          [0, 1, 2, -2, -1, 0],
          [1, 2, 3, 1, 2, 3],
        ],
      ]),
    ));
  }

  {
    const result = concatenate([fromAny([1, 2]), fromAny([3])], 0);
    asserts.assert(arrayEqual(
      result,
      fromAny([1, 2, 3]),
    ));
  }
});
