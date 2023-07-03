import { Index, Shape, T } from "./types.ts";

export const ndindex = function* (s: Shape): Iterable<Index> {
  // TODO: handle s empty
  const iterDim = function* (idx: Index, rest: Shape): Iterable<Index> {
    if (rest.length === 1) {
      for (let i = 0; i < rest[0]; i++) {
        yield [...idx, i];
      }
      return;
    }
    for (let i = 0; i < rest[0]; i++) {
      for (const result of iterDim([...idx, i], rest.slice(1))) {
        yield result;
      }
    }
  };

  for (const idx of iterDim([], s)) {
    yield idx;
  }
};
