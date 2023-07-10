import * as neural from "../../neural/mod.ts";
import * as nd from "../../ndarray/mod.ts";
import * as trees from "../../trees/mod.ts";

import { DataModule } from "./data.ts";
import { Model } from "./model.ts";

export function predict(
  dataModule: DataModule,
  model: Model,
  text: string,
  numPreds: number,
  temperature: number,
): string {
  const ids = dataModule.vectorize(text);
  let state = nd.zeros([model.numHiddens]);

  const outputs = [] as number[];

  let outID: number;
  [outID, state] = model.predictStep(ids, state, temperature);
  outputs.push(outID);

  for (let i = 0; i < numPreds; i++) {
    [outID, state] = model.predictStep(
      nd.fromAny([outID]),
      state,
      temperature,
    );
    outputs.push(outID);
  }

  return dataModule.devectorize(nd.fromArray(outputs));
}

function trainEpoch(
  dlTrain: neural.DataLoader,
  loss: neural.Loss,
  optimizer: neural.Optimizer,
  model: neural.Model,
) {
  const epochLoss = [] as number[];
  const epochAccuracy = [] as number[];
  for (const batch of dlTrain) {
    const [images, labels] = batch as [nd.T, nd.T];
    neural.resetTree(model.grads());
    const [grads, l, accuracy] = applyModel(loss, model, images, labels, true);
    model.backward(grads);
    optimizer.step(model.params(), model.grads());

    epochLoss.push(l);
    epochAccuracy.push(accuracy);
    console.log(`step loss ${l}, accuracy: ${accuracy}`);
  }
  const trainLoss = nd.meanAll(nd.fromAny(epochLoss));
  const trainAccuracy = nd.meanAll(nd.fromAny(epochAccuracy));

  return [trainLoss, trainAccuracy];
}

function evalModel(
  dlTest: neural.DataLoader,
  loss: neural.Loss,
  model: neural.Model,
) {
  const epochLoss = [] as number[];
  const epochAccuracy = [] as number[];
  for (const batch of dlTest) {
    const [images, labels] = batch as [nd.T, nd.T];
    const [_grads, l, accuracy] = applyModel(
      loss,
      model,
      images,
      labels,
      false,
    );
    epochLoss.push(l);
    epochAccuracy.push(accuracy);
  }
  const meanLoss = nd.meanAll(nd.fromAny(epochLoss));
  const meanAccuracy = nd.meanAll(nd.fromAny(epochAccuracy));

  return [meanLoss, meanAccuracy];
}

function applyModel(
  loss: neural.Loss,
  model: neural.Model,
  images: nd.T,
  labels: nd.T,
  training: boolean,
): [nd.T, number, number] {
  const yPred = model.forward(images, training);
  trees.assertIsLeaf(yPred);
  const l = loss.loss(yPred, labels);
  const grads = loss.gradient(yPred, labels);

  const accuracy = neural.computeMulticlassAccuracy(yPred, labels);

  return [grads, l, accuracy];
}

async function trainAndSave(workDir: string) {
  const batchSize = 256;
  const seqLength = 32;

  const pathTrain = `${workDir}/shakespeare_small_train.txt`;
  const pathTest = `${workDir}/shakespeare_small_test.txt`;
  const textTrain = await Deno.readTextFile(pathTrain);
  const textTest = await Deno.readTextFile(pathTest);
  const dm = new DataModule(textTrain, textTest, seqLength);
  console.log("vocab size:", dm.tokens.length);

  const dlTrain = dm.getDataLoader(batchSize, true);
  const dlTest = dm.getDataLoader(batchSize, false);

  const embedSize = 16;
  const numHiddens = 32;
  const model = new Model(dm.tokens.length, embedSize, numHiddens);

  const optimizer = neural.newGradientDescent(0.0004);
  const loss = neural.lossSoftmaxCrossEntropy;

  const [trainLoss, trainAccuracy] = trainEpoch(
    dlTrain,
    loss,
    optimizer,
    model,
  );
  const [testLoss, testAccuracy] = evalModel(
    dlTest,
    loss,
    model,
  );
  console.log(
    ` train_loss: ${trainLoss}, train_accuracy: ${trainAccuracy}, test_loss: ${testLoss}, test_accuracy: ${testAccuracy}`,
  );

  await neural.saveModelJSON(`${workDir}/model_params.json`, model);

  const predictPrompt = "Citizen";
  const predictedText = predict(dm, model, predictPrompt, 2000, 0.5);
  console.log(`Predict on "${predictPrompt}":`, predictedText);
}

if (import.meta.main) {
  await trainAndSave("./_local");
}
