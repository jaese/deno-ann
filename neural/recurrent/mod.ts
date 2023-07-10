import { asserts } from "../../deps.ts";

import * as nd from "../../ndarray/mod.ts";
import * as trees from "../../trees/mod.ts";

import * as neural from "../../neural/mod.ts";

export class SimpleRNN implements neural.Model {
  inputSize: number;
  hiddenSize: number;

  cell: neural.Model;

  constructor(
    inputSize: number,
    hiddenSize: number,
    cell: neural.Model,
  ) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;

    this.cell = cell;
  }

  forward(input: trees.T, training: boolean): trees.T {
    const [inputs, state] = input as [nd.T, nd.T];

    asserts.assertEquals(inputs.ndim(), 3);
    const [numSteps, batchSize, inputSize] = inputs.shape();
    asserts.assertEquals(inputSize, this.inputSize);

    asserts.assertEquals(state.ndim(), 2);
    asserts.assertEquals(state.shape(), [batchSize, this.hiddenSize]);
    let h = state;

    const outs = [] as nd.T[];
    for (let step = 0; step < numSteps; step++) {
      const x = inputs.get([step]);
      asserts.assertEquals(x.shape(), [batchSize, this.inputSize]);
      let out: nd.T;
      [out, h] = this.cell.forward([x, h], training) as [nd.T, nd.T];
      asserts.assertEquals(out.shape(), [batchSize, this.hiddenSize]);
      asserts.assertEquals(h.shape(), [batchSize, this.hiddenSize]);

      outs.push(nd.expandDims(out, 0));
    }

    const outsArr = nd.concatenate(outs, 0);
    asserts.assertEquals(outsArr.shape(), [
      numSteps,
      batchSize,
      this.hiddenSize,
    ]);
    return [outsArr, h];
  }
  backward(gradient: trees.T): trees.T {
    const [gradOutput, gradHOut] = gradient as [nd.T, nd.T];

    asserts.assertEquals(gradOutput.ndim(), 3);
    const [numSteps, batchSize, hiddenSize] = gradOutput.shape();
    asserts.assertEquals(hiddenSize, this.hiddenSize);
    asserts.assertEquals(gradHOut.ndim(), 2);
    asserts.assertEquals(gradHOut.shape(), [batchSize, this.hiddenSize]);

    const gradXs = [] as nd.T[];

    let h = gradHOut;

    for (let i = 0; i < numSteps; i++) {
      const step = numSteps - i - 1;
      const [gradX, gradH] = this.cell.backward([
        gradOutput.get([step]),
        h,
      ]) as [nd.T, nd.T];

      h = gradH;

      gradXs.push(nd.expandDims(gradX, 0));
    }

    gradXs.reverse();
    const gradXArr = nd.concatenate(gradXs, 0);
    asserts.assertEquals(gradXArr.shape(), [
      numSteps,
      batchSize,
      this.inputSize,
    ]);
    return [gradXArr, h];
  }

  stacks(): trees.T[][] {
    return [
      ...this.cell.stacks(),
    ];
  }
  params(): trees.T {
    return [
      this.cell.params(),
    ];
  }
  grads(): trees.T {
    return [
      this.cell.grads(),
    ];
  }
}

export class RNNCell implements neural.Model {
  inputSize: number;
  hiddenSize: number;

  wXH: nd.T;
  wHH: nd.T;
  bH: nd.T;

  activation: neural.Operation;

  gradWXH: nd.T;
  gradWHH: nd.T;
  gradBH: nd.T;

  inputs: nd.T[];
  state: nd.T[];

  constructor(
    inputSize: number,
    hiddenSize: number,
    initFn: (shape: nd.Shape) => nd.T,
  ) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;

    this.wXH = initFn([inputSize, hiddenSize]);
    this.wHH = initFn([hiddenSize, hiddenSize]);
    this.bH = nd.zeros([hiddenSize]);

    this.activation = neural.newTanh();

    this.gradWXH = nd.zeros([inputSize, hiddenSize]);
    this.gradWHH = nd.zeros([hiddenSize, hiddenSize]);
    this.gradBH = nd.zeros([hiddenSize]);

    this.inputs = [];
    this.state = [];
  }

  forward(input: trees.T, training: boolean): trees.T {
    const [inputs, state] = input as [nd.T, nd.T];

    asserts.assertEquals(inputs.ndim(), 2);
    const [batchSize, inputSize] = inputs.shape();
    asserts.assertEquals(inputSize, this.inputSize);
    asserts.assertEquals(state.ndim(), 2);
    asserts.assertEquals(state.shape(), [batchSize, this.hiddenSize]);

    if (training) {
      this.inputs.push(inputs);
      this.state.push(state);
    }

    const term1 = nd.matmul(inputs, this.wXH);
    const term2 = nd.matmul(state, this.wHH);
    const term3 = nd.add(nd.add(term1, term2), nd.expandDims(this.bH, 0));
    const outputs = this.activation.forward(term3, training) as nd.T;
    asserts.assertEquals(outputs.shape(), [batchSize, this.hiddenSize]);

    return [outputs, outputs];
  }
  backward(gradient: trees.T): [nd.T, nd.T] {
    const [gradOutput, gradStateBefore] = gradient as [nd.T, nd.T];
    asserts.assertEquals(gradOutput.ndim(), 2);
    const [batchSize, hiddenSize] = gradOutput.shape();
    asserts.assertEquals(hiddenSize, this.hiddenSize);
    asserts.assertEquals(gradStateBefore.shape(), [batchSize, hiddenSize]);

    const grad = nd.add(gradOutput, gradStateBefore);

    const g = this.activation.backward(grad) as nd.T;
    const gradX = nd.matmul(g, nd.transpose(this.wXH, [1, 0]));

    const gradState = nd.matmul(g, nd.transpose(this.wHH, [1, 0]));

    this.gradWXH.set(
      [],
      nd.add(
        this.gradWXH,
        nd.matmul(nd.transpose(this.inputs.pop()!, [1, 0]), g),
      ),
    );
    this.gradWHH.set(
      [],
      nd.add(
        this.gradWHH,
        nd.matmul(nd.transpose(this.state.pop()!, [1, 0]), g),
      ),
    );
    this.gradBH.set([], nd.add(this.gradBH, nd.sum(g, 0)));

    return [gradX, gradState];
  }

  stacks(): trees.T[][] {
    return [
      ...this.activation.stacks(),
      this.inputs,
      this.state,
    ];
  }
  params(): nd.T[] {
    return [
      this.wXH,
      this.wHH,
      this.bH,
    ];
  }
  grads(): nd.T[] {
    return [
      this.gradWXH,
      this.gradWHH,
      this.gradBH,
    ];
  }
}

export class GRUCell implements neural.Model {
  inputSize: number;
  hiddenSize: number;

  stack: [nd.T, nd.T, nd.T, nd.T, nd.T][];

  layerU: neural.Dense;
  layerR: neural.Dense;
  layerH: neural.Dense;
  layerV: neural.Dense;

  constructor(
    inputSize: number,
    hiddenSize: number,
  ) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;

    this.stack = [];

    const concatSize = this.inputSize + this.hiddenSize;
    this.layerU = new neural.Dense(
      concatSize,
      this.hiddenSize,
      neural.xavier,
      neural.newSigmoid(),
    );
    this.layerR = new neural.Dense(
      concatSize,
      this.hiddenSize,
      neural.xavier,
      neural.newSigmoid(),
    );
    this.layerH = new neural.Dense(
      concatSize,
      this.hiddenSize,
      neural.xavier,
      neural.newTanh(),
    );
    this.layerV = new neural.Dense(
      this.hiddenSize,
      this.hiddenSize,
      neural.xavier,
      null,
    );
  }

  forward(input: trees.T, training: boolean): trees.T {
    const [xIn, hIn] = input as [nd.T, nd.T];
    asserts.assertEquals(xIn.ndim(), 2);
    const [batchSize, inputSize] = xIn.shape();
    asserts.assertEquals(inputSize, this.inputSize);
    asserts.assertEquals(hIn.ndim(), 2);
    asserts.assertEquals(hIn.shape(), [batchSize, this.hiddenSize]);

    const concatenatedIn = nd.concatenate([xIn, hIn], 1);

    // reset gate
    const r = this.layerR.forward(concatenatedIn, training) as nd.T;
    asserts.assertEquals(r.shape(), [batchSize, this.hiddenSize]);

    // update gate
    const u = this.layerU.forward(
      concatenatedIn,
      training,
    ) as nd.T;
    asserts.assertEquals(u.shape(), [batchSize, this.hiddenSize]);

    // new state
    const hReset = nd.mul(r, hIn);
    asserts.assertEquals(hReset.shape(), [batchSize, this.hiddenSize]);

    const resetConcat = nd.concatenate([xIn, hReset], 1);

    // H new
    const hBar = this.layerH.forward(
      resetConcat,
      training,
    ) as nd.T;
    asserts.assertEquals(hBar.shape(), [batchSize, this.hiddenSize]);

    const hOut = nd.add(
      nd.mul(u, hIn),
      nd.mul(nd.sub(nd.ones(u.shape()), u), hBar),
    );
    asserts.assertEquals(hOut.shape(), [batchSize, this.hiddenSize]);

    const xOut = this.layerV.forward(hOut, training) as nd.T;
    asserts.assertEquals(xOut.shape(), [batchSize, this.hiddenSize]);

    if (training) {
      this.stack.push([xIn, hIn, u, r, hBar]);
    }

    return [xOut, hOut];
  }
  backward(gradient: trees.T): trees.T {
    const [gradXOut, gradHOut2] = gradient as [nd.T, nd.T];
    const [batchSize, hiddenSize] = gradXOut.shape();
    asserts.assertEquals(hiddenSize, this.hiddenSize);

    const [_xIn, hIn, u, r, hBar] = this.stack.pop()!;

    const gradHOut1 = this.layerV.backward(gradXOut) as nd.T;

    const gradHOut = nd.add(gradHOut1, gradHOut2);

    const gradU1 = nd.mul(hIn, gradHOut);
    const gradHIn3 = nd.mul(u, gradHOut);
    asserts.assertEquals(gradHIn3.shape(), [batchSize, this.hiddenSize]);
    const gradHBar = nd.mul(nd.sub(nd.ones(u.shape()), u), gradHOut);
    const gradU2 = nd.scale(nd.mul(hBar, gradHOut), -1);

    const gradU = nd.add(gradU1, gradU2);

    const gradResetConcat = this.layerH.backward(gradHBar) as nd.T;

    const gradResetConcatT = nd.swapaxes(gradResetConcat, 0, 1);
    asserts.assertEquals(gradResetConcatT.shape(), [
      this.inputSize + this.hiddenSize,
      batchSize,
    ]);
    const gradXIn2T = nd.slice(gradResetConcatT, 0, this.inputSize);
    const gradHResetT = nd.slice(
      gradResetConcatT,
      this.inputSize,
      this.inputSize + this.hiddenSize,
    );

    const gradXIn2 = nd.swapaxes(gradXIn2T, 0, 1);
    asserts.assertEquals(gradXIn2.shape(), [batchSize, this.inputSize]);
    const gradHReset = nd.swapaxes(gradHResetT, 0, 1);
    asserts.assertEquals(gradHReset.shape(), [batchSize, this.hiddenSize]);

    const gradR = nd.mul(gradHReset, hIn);
    const gradHIn2 = nd.mul(r, gradHReset);
    asserts.assertEquals(gradHIn2.shape(), [batchSize, this.hiddenSize]);

    const gradConcatenatedIn2 = this.layerU.backward(gradU) as nd.T;
    const gradConcatenatedIn1 = this.layerR.backward(gradR) as nd.T;

    const gradConcatenatedIn = nd.add(gradConcatenatedIn1, gradConcatenatedIn2);

    const gradConcatenatedInT = nd.swapaxes(gradConcatenatedIn, 0, 1);
    asserts.assertEquals(gradConcatenatedInT.shape(), [
      this.inputSize + this.hiddenSize,
      batchSize,
    ]);

    const gradXIn1T = nd.slice(gradConcatenatedInT, 0, this.inputSize);
    const gradXIn1 = nd.swapaxes(gradXIn1T, 0, 1);
    asserts.assertEquals(gradXIn1.shape(), [batchSize, this.inputSize]);
    const gradHIn1T = nd.slice(
      gradConcatenatedInT,
      this.inputSize,
      this.inputSize + this.hiddenSize,
    );
    const gradHIn1 = nd.swapaxes(gradHIn1T, 0, 1);
    asserts.assertEquals(gradHIn1.shape(), [batchSize, this.hiddenSize]);

    const gradHIn = nd.add(nd.add(gradHIn1, gradHIn2), gradHIn3);
    const gradXIn = nd.add(gradXIn1, gradXIn2);

    return [gradXIn, gradHIn];
  }

  stacks(): trees.T[][] {
    return [
      ...this.layerV.stacks(),
      ...this.layerU.stacks(),
      ...this.layerR.stacks(),
      ...this.layerH.stacks(),
    ];
  }
  params(): trees.T {
    return [
      this.layerV.params(),
      this.layerU.params(),
      this.layerR.params(),
      this.layerH.params(),
    ];
  }
  grads(): trees.T {
    return [
      this.layerV.grads(),
      this.layerU.grads(),
      this.layerR.grads(),
      this.layerH.grads(),
    ];
  }
}
