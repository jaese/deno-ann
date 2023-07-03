import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";

import { isLeaf, T } from "./types.ts";

export function jsonEncode(t: T): string {
  const x = convertForEncode(t);
  return JSON.stringify(x);
}

function convertForEncode(t: T): any {
  if (isLeaf(t)) {
    const arr = t as nd.T;
    return {
      "@type": "ndarray",
      shape: arr.shape(),
      buffer: Array.from(arr.buffer()),
    };
  } else if (t instanceof Array) {
    return t.map(convertForEncode);
  } else if (t instanceof Map) {
    const converted = {} as { [id: string]: any };
    for (const [key, item] of t.entries()) {
      const itemConverted = convertForEncode(item);
      converted[key] = itemConverted;
    }
    return converted;
  } else {
    throw Error(t);
  }
}

export function jsonDecode(bs: string): T {
  const decoded = JSON.parse(bs);
  return convertFromDecode(decoded);
}

export function convertFromDecode(x: any): T {
  if ("@type" in x) {
    asserts.assertEquals(x["@type"], "ndarray");
    return nd.make(
      x.shape,
      new Float32Array(x.buffer),
    );
  } else if (x instanceof Array) {
    return x.map(convertFromDecode);
  } else {
    const entries = [] as [string, any][];
    for (const [key, item] of Object.entries(x)) {
      entries.push([key, convertFromDecode(item)]);
    }
    return new Map(entries);
  }
}
