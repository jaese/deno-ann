import { asserts } from "../../deps.ts";

import * as numerical from "../../numerical/mod.ts";
import * as neural from "../../neural/mod.ts";
import * as recurrent from "../../neural/recurrent/mod.ts";
import * as nd from "../../ndarray/mod.ts";
import * as trees from "../../trees/mod.ts";

export class Model implements neural.Model {
  vocabSize: number;
  embedSize: number;
  numHiddens: number;

  layerEmbedding: neural.Embedding;
  layerRNN: recurrent.SimpleRNN;
  layerDense: neural.Dense;

  constructor(vocabSize: number, embedSize: number, numHiddens: number) {
    this.vocabSize = vocabSize;
    this.embedSize = embedSize;
    this.numHiddens = numHiddens;

    this.layerEmbedding = new neural.Embedding(this.vocabSize, this.embedSize);
    this.layerRNN = new recurrent.SimpleRNN(
      this.embedSize,
      this.numHiddens,
      new neural.Sequential([
        [
          "rnn_cell",
          new recurrent.RNNCell(this.embedSize, this.numHiddens, neural.xavier),
        ],
        ["dropout", new neural.Dropout(0.2)],
      ]),
    );
    this.layerDense = new neural.Dense(
      this.numHiddens,
      this.vocabSize,
      neural.xavier,
      null,
    );
  }

  forward(input: trees.T, training: boolean): trees.T {
    trees.assertIsLeaf(input);
    const [batchSize, _seqLength] = input.shape();

    const [y, _h] = this.forwardWithState(
      input,
      nd.zeros([batchSize, this.numHiddens]),
      training,
    );
    return y;
  }
  forwardWithState(x: nd.T, h: nd.T, training: boolean): [nd.T, nd.T] {
    asserts.assertEquals(x.ndim(), 2);
    const [batchSize, seqLength] = x.shape();
    asserts.assertEquals(h.ndim(), 2);
    asserts.assertEquals(h.shape(), [batchSize, this.numHiddens]);

    // Embedding layer
    x = nd.reshape(x, [batchSize * seqLength]);
    x = this.layerEmbedding.forward(x, training) as nd.T;
    asserts.assertEquals(x.ndim(), 2);

    // RNN layer
    x = nd.reshape(x, [batchSize, seqLength, this.embedSize]);
    x = nd.transpose(x, [1, 0, 2]);
    let nextH: nd.T;
    [x, nextH] = this.layerRNN.forward([x, h], training) as [nd.T, nd.T];
    asserts.assertEquals(x.ndim(), 3);
    asserts.assertEquals(x.shape(), [seqLength, batchSize, this.numHiddens]);
    x = nd.transpose(x, [1, 0, 2]);
    x = nd.reshape(x, [batchSize * seqLength, this.numHiddens]);

    // FC output layer
    x = this.layerDense.forward(x, training) as nd.T;
    asserts.assertEquals(x.ndim(), 2);
    x = nd.reshape(x, [batchSize, seqLength, this.vocabSize]);

    return [x, nextH];
  }
  backward(gradient: trees.T): trees.T {
    trees.assertIsLeaf(gradient);
    let g = gradient;
    asserts.assertEquals(g.ndim(), 3);
    const [batchSize, seqLength, vocabSize] = g.shape();
    asserts.assertEquals(vocabSize, this.vocabSize);

    g = nd.reshape(g, [batchSize * seqLength, this.vocabSize]);
    g = this.layerDense.backward(g) as nd.T;
    g = nd.reshape(g, [batchSize, seqLength, this.numHiddens]);
    g = nd.transpose(g, [1, 0, 2]);

    let _gradH: nd.T;
    [g, _gradH] = this.layerRNN.backward([
      g,
      nd.zeros([batchSize, this.numHiddens]),
    ]) as [nd.T, nd.T];
    g = nd.transpose(g, [1, 0, 2]);
    g = nd.reshape(g, [batchSize * seqLength, this.embedSize]);

    g = this.layerEmbedding.backward(g) as nd.T;

    g = nd.reshape(g, [batchSize, seqLength]);

    return g;
  }

  stacks(): trees.T[][] {
    return [
      ...this.layerEmbedding.stacks(),
      ...this.layerRNN.stacks(),
      ...this.layerDense.stacks(),
    ];
  }
  params(): trees.T {
    return [
      this.layerEmbedding.params(),
      this.layerRNN.params(),
      this.layerDense.params(),
    ];
  }
  grads(): trees.T {
    return [
      this.layerEmbedding.grads(),
      this.layerRNN.grads(),
      this.layerDense.grads(),
    ];
  }

  predictStep(input: nd.T, state: nd.T, temperature: number): [number, nd.T] {
    const [seqLength] = input.shape();

    const [output, nextState] = this.forwardWithState(
      nd.expandDims(input, 0),
      nd.expandDims(state, 0),
      false,
    );
    asserts.assertEquals(output.shape(), [1, seqLength, this.vocabSize]);

    const outLogits = output.get([0, seqLength - 1]);
    outLogits.set([0], nd.fromAny(-Infinity));

    const probas = neural.softmax(nd.scale(outLogits, 1 / temperature));
    const outID = numerical.getRandomIntProbas(nd.toArray(probas));
    return [outID, nd.squeeze(nextState, 0)];
  }
}
