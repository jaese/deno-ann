import * as nd from "../ndarray/mod.ts";
import * as trees from "../trees/mod.ts";

export interface Operation {
  forward(input: trees.T, training: boolean): trees.T;
  backward(gradient: trees.T): trees.T;

  stacks(): trees.T[][];
}

export interface Model extends Operation {
  params(): trees.T;
  grads(): trees.T;
}

export const isModel = function (a: Operation): a is Model {
  return ("params" in a) && ("grads" in a);
};

export interface Loss {
  loss(predicted: nd.T, target: nd.T): number;
  gradient(predicted: nd.T, target: nd.T): nd.T;
}

export interface Optimizer {
  step(params: trees.T, grads: trees.T): void;
  epoch(): void;
}
