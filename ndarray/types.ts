export type Shape = number[];
export type Index = number[];

export interface T {
  shape(): Shape;
  ndim(): number;
  buffer(): Float32Array;

  item(): number;

  get(idx: Index): T;
  set(idx: Index, v: T): void;

  reshape(shape: Shape): void;

  toString(): string;
}
