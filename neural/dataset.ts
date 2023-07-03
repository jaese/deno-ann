import { shuffle } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";
import * as trees from "../trees/mod.ts";

export interface Dataset {
  len(): number;
  get(idx: number): trees.T;
}

export class ArrayDataset implements Dataset {
  xs: trees.T[];

  constructor(xs: trees.T[]) {
    this.xs = xs;
  }

  len(): number {
    return this.xs.length;
  }
  get(idx: number): trees.T {
    return this.xs[idx];
  }
}

export class TreeDataset implements Dataset {
  data: trees.T;
  size: number;

  constructor(data: trees.T) {
    this.data = data;
    this.size = trees.plurality(data);
  }

  len(): number {
    return this.size;
  }
  get(idx: number): trees.T {
    return trees.map(([x]) => x.get([idx]), [this.data]);
  }
}

export type DataLoader = Iterable<trees.T>;

export function dataLoaderFromDataset(ds: Dataset): DataLoader {
  return {
    *[Symbol.iterator]() {
      for (let i = 0; i < ds.len(); i++) {
        yield ds.get(i);
      }
    },
  };
}

export function dataLoaderShuffle(
  dl: DataLoader,
  bufferSize: number,
): DataLoader {
  const batchedDL = dataLoaderBatch(dl, bufferSize, true);

  const shuffledDL = {
    *[Symbol.iterator]() {
      for (const batch of batchedDL) {
        const permutation = getRandomPermutation(bufferSize);
        yield trees.map(([x]) => permuateArray(x, permutation), [batch]);
      }
    },
  } as DataLoader;

  return dataLoaderUnbatch(shuffledDL);
}

export function dataLoaderBatch(
  dl: DataLoader,
  batchSize: number,
  dropRemaining: boolean,
): DataLoader {
  const f = function* (): Iterator<trees.T> {
    let batch = [] as trees.T[];
    for (const item of dl) {
      if (batch.length < batchSize) {
        batch.push(item);
      } else {
        const concatenatedBatch = trees.map(
          (xs) => nd.concatenate(xs.map((x) => nd.expandDims(x, 0)), 0),
          batch,
        );
        yield concatenatedBatch;
        batch = [item];
      }
    }
    if (!dropRemaining) {
      if (batch.length !== 0) {
        const concatenatedBatch = trees.map(
          (xs) => nd.concatenate(xs.map((x) => nd.expandDims(x, 0)), 0),
          batch,
        );
        yield concatenatedBatch;
      }
    }
  };

  return {
    [Symbol.iterator]() {
      return f();
    },
  };
}

export function dataLoaderUnbatch(dl: DataLoader): DataLoader {
  const f = function* (): Iterator<trees.T> {
    for (const batch of dl) {
      const n = trees.plurality(batch);
      for (let i = 0; i < n; i++) {
        const item = trees.map(([x]) => x.get([i]), [batch]);
        yield item;
      }
    }
  };
  return {
    [Symbol.iterator]() {
      return f();
    },
  };
}

function getRandomPermutation(n: number): number[] {
  const permutation = shuffle.default(new Array(n).fill(0).map((_, i) => i));
  return permutation;
}

function permuateArray(x: nd.T, permutation: number[]): nd.T {
  const result = nd.zeros(x.shape());
  const n = x.shape()[0];

  for (let i = 0; i < n; i++) {
    const p = permutation[i];
    result.set([p], x.get([i]));
  }
  return result;
}
