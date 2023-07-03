import { asserts } from "../deps.ts";

import { isclose } from "./mod.ts";

Deno.test("isclose", () => {
  asserts.assert(isclose(3, 3.00001));
  asserts.assert(!isclose(3, 4));
});
