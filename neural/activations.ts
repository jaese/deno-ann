import * as nd from "../ndarray/mod.ts";
import * as trees from "../trees/mod.ts";

import { Loss, Operation, Optimizer } from "./types.ts";

export function newSigmoid(): Operation {
  const l = {
    sigmoids: [] as trees.T[],

    forward(input: trees.T, training: boolean): trees.T {
      const output = trees.map(([x]) => nd.apply(x, sigmoid), [input]);
      if (training) {
        this.sigmoids.push(output);
      }
      return output;
    },

    backward(gradient: trees.T): trees.T {
      return trees.map(([g, s]) =>
        nd.combine(
          g,
          s,
          (grad: number, sig: number) => sig * (1 - sig) * grad,
        ), [gradient, this.sigmoids.pop()!]);
    },

    stacks(): trees.T[][] {
      return [this.sigmoids];
    },
  };

  return l;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function tanh(x: number): number {
  if (x < -100) {
    return -1;
  } else if (x > 100) {
    return 1;
  }

  const em2x = Math.exp(-2 * x);
  return (1 - em2x) / (1 + em2x);
}

// Updated upstream
export function newTanh(): Operation {
  const l = {
    tanh: [] as trees.T[],

    forward(input: trees.T, training: boolean): trees.T {
      const output = trees.map(([xs]) => nd.apply(xs, tanh), [input]);
      if (training) {
        this.tanh.push(output);
      }
      return output;
    },

    backward(gradient: trees.T): trees.T {
      return trees.map(([t, g]) =>
        nd.combine(
          t,
          g,
          (tanh: number, grad: number) => (1 - tanh ** 2) * grad,
        ), [this.tanh.pop()!, gradient]);
    },

    stacks(): trees.T[][] {
      return [this.tanh];
    },
  };

  return l;
}

export function newRelu(): Operation {
  const l = {
    input: [] as trees.T[],

    forward(input: trees.T, training: boolean): trees.T {
      if (training) {
        this.input.push(input);
      }
      return trees.map(([xs]) => nd.apply(xs, relu), [input]);
    },

    backward(gradient: trees.T): trees.T {
      return trees.map(([xs, g]) =>
        nd.combine(
          xs,
          g,
          (input: number, grad: number) => input > 0 ? grad : 0,
        ), [this.input.pop()!, gradient]);
    },

    stacks(): trees.T[][] {
      return [this.input];
    },
  };

  return l;
}

export function relu(x: number): number {
  return Math.max(0, x);
}
