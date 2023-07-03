import * as nd from "../ndarray/mod.ts";
import * as trees from "../trees/mod.ts";

import { Optimizer } from "./types.ts";

export function newGradientDescent(learningRate: number): Optimizer {
  const o = {
    lr: learningRate,
    step(params: trees.T, grads: trees.T): void {
      const updateParam = (param: nd.T, grad: nd.T) => {
        param.set(
          [],
          nd.add(param, nd.apply(grad, (g: number) => -g * this.lr)),
        );
      };
      trees.forEach((xs) => updateParam(xs[0], xs[1]), [
        params,
        grads,
      ]);
    },
    epoch(): void {},
  };

  return o;
}

export function newMomentum(
  lr: number,
  finalLR: number,
  maxEpochs: number,
  momentum: number,
): Optimizer {
  const decayPerEpoch = Math.pow(finalLR / lr, 1 / (maxEpochs - 1));

  const o = {
    lr: lr,
    decayPerEpoch: decayPerEpoch,
    mo: momentum,
    updates: null as null | trees.T,

    step(params: trees.T, grads: trees.T): void {
      if (this.updates === null) {
        this.updates = trees.map(
          ([xs]) => nd.zeros(xs.shape()),
          [grads],
        );
      }

      const f = ([update, param, grad]: nd.T[]) => {
        update.set(
          [],
          nd.combine(
            update,
            grad,
            (u, g) => this.mo * u + (1 - this.mo) * g,
          ),
        );

        param.set(
          [],
          nd.combine(
            param,
            update,
            (p, u) => p - this.lr * u,
          ),
        );
        return nd.zeros(update.shape());
      };

      trees.map(f, [this.updates, params, grads]);
    },

    epoch(): void {
      this.lr = this.lr * this.decayPerEpoch;
    },
  };

  return o;
}
