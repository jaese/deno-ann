import { asserts } from "../deps.ts";

import * as nd from "../ndarray/mod.ts";
import * as trees from "../trees/mod.ts";

import { randomUniformBothWays } from "./inits.ts";
import { isModel, Model, Operation } from "./types.ts";

export class Dense implements Model {
  w: nd.T;
  b: nd.T;

  wGrad: nd.T;
  bGrad: nd.T;

  inputDim: number;
  outputDim: number;
  activation: Operation | null;

  input: nd.T[];

  constructor(
    inputDim: number,
    outputDim: number,
    initFunc: (shape: nd.Shape) => nd.T,
    activation: Operation | null,
  ) {
    this.w = initFunc([inputDim, outputDim]);
    this.b = nd.zeros([outputDim]);

    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.activation = activation;

    this.wGrad = nd.zeros([inputDim, outputDim]);
    this.bGrad = nd.zeros([outputDim]);

    this.input = [];
  }
  forward(input: trees.T, training: boolean): trees.T {
    trees.assertIsLeaf(input);
    asserts.assertEquals(input.ndim(), 2);
    const [batchSize, _inputDim] = input.shape();

    if (training) {
      this.input.push(input);
    }

    let y = nd.add(
      nd.matmul(
        input,
        this.w,
      ),
      nd.expandDims(this.b, 0),
    );
    asserts.assertEquals(y.shape(), [batchSize, this.outputDim]);

    if (this.activation !== null) {
      y = this.activation.forward(y, training) as nd.T;
    }

    return y;
  }

  backward(gradient: trees.T): trees.T {
    trees.assertIsLeaf(gradient);
    const [batchSize, outputDim] = gradient.shape();
    asserts.assertEquals(outputDim, this.outputDim);

    const input = this.input.pop()!;
    asserts.assertEquals(input.shape(), [
      batchSize,
      this.inputDim,
    ]);

    let grad = gradient;
    if (this.activation !== null) {
      grad = this.activation.backward(gradient) as nd.T;
    }

    this.bGrad.set([], nd.add(this.bGrad, nd.sum(grad, 0)));
    asserts.assertEquals(this.bGrad.shape(), [this.outputDim]);

    this.wGrad.set(
      [],
      nd.add(
        this.wGrad,
        nd.matmul(
          nd.transpose(input, [1, 0]),
          grad,
        ),
      ),
    );
    asserts.assertEquals(this.wGrad.shape(), [this.inputDim, this.outputDim]);

    const xGrad = nd.matmul(grad, nd.transpose(this.w, [1, 0]));
    asserts.assertEquals(xGrad.shape(), [batchSize, this.inputDim]);

    return xGrad;
  }
  stacks(): trees.T[][] {
    return [
      this.input,
      ...(this.activation !== null ? this.activation.stacks() : []),
    ];
  }
  params(): trees.T {
    return [this.w, this.b];
  }
  grads(): trees.T {
    return [this.wGrad, this.bGrad];
  }
}

export interface Conv1D extends Model {
  kernel: nd.T;
}

export const newConv1D = (
  kernelSize: number,
  initFn: (shape: nd.Shape) => nd.T,
): Conv1D => {
  asserts.assertEquals(kernelSize % 2, 1);
  const kernel = initFn([kernelSize]);

  const l = {
    kernel: kernel,
    kernelSize: kernelSize,

    input: [] as nd.T[],

    gradKernel: nd.zeros([kernelSize]),

    forward(input: trees.T, training: boolean): trees.T {
      trees.assertIsLeaf(input);
      asserts.assertEquals(input.ndim(), 1);

      if (training) {
        this.input.push(input);
      }
      const inputPadded = nd.pad1D(input, Math.floor(this.kernelSize / 2));

      const output = nd.convolveValid(inputPadded, this.kernel);
      asserts.assertEquals(output.shape()[0], input.shape()[0]);

      return output;
    },
    backward(gradient: trees.T): trees.T {
      trees.assertIsLeaf(gradient);

      const input = this.input.pop()!;

      const gradientPadded = nd.pad1D(
        gradient,
        Math.floor(this.kernelSize / 2),
      );

      this.gradKernel.set(
        [],
        nd.add(
          this.gradKernel,
          nd.convolveValid(
            gradientPadded,
            nd.flip1D(input),
          ),
        ),
      );

      const gradX = nd.convolveValid(gradientPadded, nd.flip1D(this.kernel));
      return gradX;
    },
    stacks(): trees.T[][] {
      return [this.input];
    },
    params(): trees.T {
      return [this.kernel];
    },
    grads(): trees.T {
      return [this.gradKernel];
    },
  };

  return l;
};

export class Sequential implements Model {
  layerList: [string, Operation][];

  constructor(layerList: [string, Operation][]) {
    this.layerList = layerList;
  }

  forward(input: trees.T, training: boolean): trees.T {
    let x = input;
    for (const layer of this.layerList) {
      x = layer[1].forward(x, training);
    }
    return x;
  }
  backward(gradient: trees.T): trees.T {
    let g = gradient;
    for (const layer of this.layerList.toReversed()) {
      g = layer[1].backward(g);
    }
    return g;
  }
  stacks(): trees.T[][] {
    const stacks = [] as trees.T[][];
    for (const layer of this.layerList) {
      stacks.push(...layer[1].stacks());
    }
    return stacks;
  }
  params(): trees.T {
    return new Map(
      this.layerList.filter((layer) => isModel(layer[1])).map((
        layer,
      ) => [layer[0], (layer[1] as Model).params()]) as [string, trees.T][],
    );
  }
  grads(): trees.T {
    return new Map(
      this.layerList.filter((layer) => isModel(layer[1])).map((
        layer,
      ) => [layer[0], (layer[1] as Model).grads()]) as [string, trees.T][],
    );
  }
  getLayer(name: string): Model {
    for (const [layerName, layer] of this.layerList) {
      if (layerName === name) {
        return layer as Model;
      }
    }
    throw new Error(name);
  }
}

export class Dropout implements Operation {
  p: number;

  mask: trees.T[];

  constructor(p: number) {
    this.p = p;

    this.mask = [];
  }

  forward(input: trees.T, training: boolean): trees.T {
    const scale = 1 - this.p;
    if (!training) {
      return trees.map(([xs]) => nd.apply(xs, (x) => x * scale), [input]);
    } else {
      const mask = trees.map(
        ([xs]) => nd.apply(xs, (_) => Math.random() < this.p ? 0 : 1),
        [input],
      );
      this.mask.push(mask);
      return trees.map(([xs, m]) => nd.mul(xs, m), [input, mask]);
    }
  }
  backward(gradient: trees.T): trees.T {
    const mask = this.mask.pop()!;
    return trees.map(([g, m]) => nd.mul(g, m), [gradient, mask]);
  }
  stacks(): trees.T[][] {
    return [this.mask];
  }
}

export class Embedding implements Model {
  numEmbeddings: number;
  embeddingDim: number;

  embeddings: nd.T;
  grad: nd.T;

  inputID: nd.T[];

  constructor(numEmbeddings: number, embeddingDim: number) {
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;

    this.embeddings = randomUniformBothWays([numEmbeddings, embeddingDim]);
    this.grad = nd.zeros(this.embeddings.shape());
    this.inputID = [];
  }

  forward(input: trees.T, training: boolean): trees.T {
    trees.assertIsLeaf(input);
    asserts.assertEquals(input.ndim(), 1);
    const [batchSize] = input.shape();

    if (training) {
      this.inputID.push(input);
    }

    const output = nd.zeros([batchSize, this.embeddingDim]);
    for (let i = 0; i < batchSize; i++) {
      const tokenID = input.get([i]).item();
      asserts.assert(Number.isInteger(tokenID));
      asserts.assert(tokenID < this.numEmbeddings);
      output.set([i], this.embeddings.get([tokenID]));
    }
    return output;
  }

  backward(gradient: trees.T): trees.T {
    trees.assertIsLeaf(gradient);
    asserts.assertEquals(gradient.ndim(), 2);
    const [batchSize, embeddingDim] = gradient.shape();
    asserts.assertEquals(embeddingDim, this.embeddingDim);

    const inputID = this.inputID.pop()!;

    for (let i = 0; i < batchSize; i++) {
      const id = inputID.get([i]).item();
      this.grad.get([id]).set(
        [],
        nd.add(this.grad.get([id]), gradient.get([i])),
      );
    }

    // gradients are not passed.
    return nd.zeros([batchSize]);
  }

  stacks(): trees.T[][] {
    return [
      this.inputID,
    ];
  }
  params(): trees.T {
    return [
      this.embeddings,
    ];
  }
  grads(): trees.T {
    return [
      this.grad,
    ];
  }
}
