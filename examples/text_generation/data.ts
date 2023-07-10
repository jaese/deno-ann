import { asserts } from "../../deps.ts";

import * as neural from "../../neural/mod.ts";
import * as nd from "../../ndarray/mod.ts";
import * as trees from "../../trees/mod.ts";

export async function createDataModule(
  pathTrain: string,
  pathTest: string,
  seqLength: number,
): Promise<DataModule> {
  const textTrain = await Deno.readTextFile(pathTrain);
  const textTest = await Deno.readTextFile(pathTest);

  const dm = new DataModule(textTrain, textTest, seqLength);

  return dm;
}

export class DataModule {
  textTrain: string;
  textTest: string;
  seqLength: number;

  tokens: string[];
  tokenToIndex: Map<string, number>;

  constructor(
    textTrain: string,
    textTest: string,
    seqLength: number,
  ) {
    this.textTrain = textTrain;
    this.textTest = textTest;
    this.seqLength = seqLength;

    const tokens = [
      "<UNK>",
      "-",
      "\n",
      " ",
      " -",
      "!",
      "?",
      "'",
      ",",
      ":",
      ".",
    ];
    for (let c = 0x41; c <= 0x5A; c++) {
      tokens.push(String.fromCharCode(c));
    }
    for (let c = 0x61; c <= 0x7A; c++) {
      tokens.push(String.fromCharCode(c));
    }

    const tokenToIndex = new Map<string, number>();
    for (let i = 0; i < tokens.length; i++) {
      tokenToIndex.set(tokens[i], i);
    }

    this.tokens = tokens;
    this.tokenToIndex = tokenToIndex;
  }

  lookupTokenID(token: string): number {
    const id = this.tokenToIndex.get(token);
    if (id === undefined) {
      return 0;
    } else {
      return id;
    }
  }

  vectorize(text: string): nd.T {
    const ids = [] as number[];
    for (const c of text) {
      ids.push(this.lookupTokenID(c));
    }
    const arr = nd.fromArray(ids);
    return arr;
  }

  devectorize(arr: nd.T): string {
    asserts.assertEquals(arr.ndim(), 1);
    const n = arr.shape()[0];
    const chars = [] as string[];
    for (let i = 0; i < n; i++) {
      chars.push(this.tokens[arr.get([i]).item()]);
    }
    return chars.join("");
  }

  getDataLoader(batchSize: number, training: boolean): neural.DataLoader {
    const text = training ? this.textTrain : this.textTest;
    const ids = this.vectorize(text);

    const numTokens = ids.shape()[0];

    const self = this;
    const dl = {
      *[Symbol.iterator](): Iterator<trees.T> {
        for (let i = 0; i < numTokens - self.seqLength - 1; i++) {
          const x = nd.slice(
            ids,
            i,
            i + self.seqLength,
          );
          const y = nd.slice(
            ids,
            i + 1,
            i + self.seqLength + 1,
          );

          const item = [x, y];
          yield item;
        }
      },
    };

    const shuffled = neural.dataLoaderShuffle(dl, 1024);
    const batched = neural.dataLoaderBatch(shuffled, batchSize, false);

    return batched;
  }
}
