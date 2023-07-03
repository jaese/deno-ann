import * as nd from "../../ndarray/mod.ts";
import * as neural from "../../neural/mod.ts";
import * as trees from "../../trees/mod.ts";

import { getDatasets } from "./dataset.ts";

function createModel(): neural.Model {
  const model = new neural.Sequential([
    ["dense1", new neural.Dense(784, 30, neural.xavier, neural.newTanh())],
    ["dropout1", new neural.Dropout(0.2)],
    ["dense2", new neural.Dense(30, 10, neural.xavier, neural.newTanh())],
    ["dropout2", new neural.Dropout(0.2)],
    ["dense3", new neural.Dense(10, 10, neural.xavier, null)],
  ]);
  return model;
}

async function trainAndEvaluate(
  workDir: string,
  batchSize: number,
): Promise<void> {
  const [[xTrain, yTrain], [xTest, yTest]] = await getDatasets("./_local");

  const numEpochs = 3;

  const loss = neural.lossSoftmaxCrossEntropy;
  // const optimizer = neural.newGradientDescent(0.01);
  const optimizer = neural.newMomentum(0.005, 0.001, numEpochs, 0.9);
  const model = createModel();

  const dsTrain = new neural.TreeDataset([xTrain, yTrain]);
  let epoch: number;
  for (epoch = 0; epoch < numEpochs; epoch++) {
    const [trainLoss, trainAccuracy] = trainEpoch(
      loss,
      optimizer,
      model,
      dsTrain,
      batchSize,
    );
    const [_, testLoss, testAccuracy] = applyModel(
      loss,
      model,
      xTest,
      yTest,
      false,
    );

    console.log(
      `epoch: ${epoch}, train_loss: ${trainLoss}, train_accuracy: ${trainAccuracy}, test_loss: ${testLoss}, test_accuracy: ${testAccuracy}`,
    );
  }

  saveMNISTModel(workDir, epoch, model);
}

function trainEpoch(
  loss: neural.Loss,
  optimizer: neural.Optimizer,
  model: neural.Model,
  dsTrain: neural.Dataset,
  batchSize: number,
): [number, number] {
  const dlTrain = neural.dataLoaderBatch(
    neural.dataLoaderShuffle(
      neural.dataLoaderFromDataset(dsTrain),
      1024,
    ),
    batchSize,
    false,
  );

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
  }
  const trainLoss = nd.meanAll(nd.fromAny(epochLoss));
  const trainAccuracy = nd.meanAll(nd.fromAny(epochAccuracy));

  return [trainLoss, trainAccuracy];
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

async function saveMNISTModel(
  workDir: string,
  iter: number,
  model: neural.Model,
): Promise<void> {
  const filepath = `${workDir}/params_${iter}.json`;
  console.log("Saving params to", filepath);
  await neural.saveModelJSON(filepath, model);
}

if (import.meta.main) {
  await trainAndEvaluate("_local", 128);
}
