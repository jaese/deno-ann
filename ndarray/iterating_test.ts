import { asserts } from "../deps.ts";

import { ndindex } from "./iterating.ts";

Deno.test("ndindex", () => {
  const result = Array.from(ndindex([3, 2, 1]));
  asserts.assertEquals(
    result,
    [
      [0, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 0],
      [2, 0, 0],
      [2, 1, 0],
    ],
  );
});
