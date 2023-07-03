import { asserts, io } from "../../deps.ts";
import { struct } from "./deps.ts";

import * as nd from "../../ndarray/mod.ts";

export async function getDatasets(
  datasetDir: string,
): Promise<[[nd.T, nd.T], [nd.T, nd.T]]> {
  const [[trainImages, trainLabels], [testImages, testLabels]] = await loadData(
    datasetDir,
  );

  const avg = nd.sumAll(trainImages) / nd.size(trainImages);
  const xTrain = preprocessX(trainImages, avg);
  const xTest = preprocessX(testImages, avg);

  const yTrain = nd.fromAny(trainLabels);
  asserts.assertEquals(yTrain.ndim(), 1);
  const yTest = nd.fromAny(testLabels);
  asserts.assertEquals(yTest.ndim(), 1);

  return [[xTrain, yTrain], [xTest, yTest]];
}

async function readImagesAndLabels(
  imagesFilepath: string,
  labelsFilepath: string,
): Promise<[number[], nd.T]> {
  const labels = [] as number[];

  const labelsFile = await Deno.open(labelsFilepath, { read: true });
  try {
    const reader = new io.BufReader(labelsFile);
    const [magic, _size] = struct.Struct.unpack(
      ">II",
      (await reader.readFull(new Uint8Array(8)))!,
    );
    asserts.assertEquals(magic, 2049);

    while (true) {
      const b = await reader.readByte();
      if (b === null) {
        break;
      }
      labels.push(b);
    }
  } finally {
    labelsFile.close();
  }

  let images: nd.T;
  const imagesFile = await Deno.open(imagesFilepath, { read: true });
  try {
    const reader = new io.BufReader(imagesFile);
    const header = struct.Struct.unpack(
      ">IIII",
      (await reader.readFull(new Uint8Array(16)))!,
    );
    const magic = header[0] as number;
    const size = header[1] as number;
    const rows = header[2] as number;
    const cols = header[3] as number;
    asserts.assertEquals(magic, 2051);

    images = nd.zeros([size, rows, cols]);

    const buffer = new Uint8Array(rows * cols);
    for (let i = 0; i < size; i++) {
      await reader.readFull(buffer);
      images.get([i]).buffer().set(buffer);
    }
  } finally {
    imagesFile.close();
  }

  return [labels, images];
}

async function loadData(
  datasetPath: string,
): Promise<[[nd.T, number[]], [nd.T, number[]]]> {
  const [trainLabels, trainImages] = await readImagesAndLabels(
    `${datasetPath}/train-images.idx3-ubyte`,
    `${datasetPath}/train-labels.idx1-ubyte`,
  );
  const [testLabels, testImages] = await readImagesAndLabels(
    `${datasetPath}/t10k-images.idx3-ubyte`,
    `${datasetPath}/t10k-labels.idx1-ubyte`,
  );

  return [[trainImages, trainLabels], [testImages, testLabels]];
}

function preprocessX(images: nd.T, avg: number): nd.T {
  asserts.assertEquals(images.ndim(), 3);
  const n = images.shape()[0];
  let X = nd.reshape(images, [n, nd.size(images) / n]);

  X = nd.scale(nd.sub(X, nd.fromAny([[avg]])), 1 / 255);
  asserts.assertEquals(X.shape()[1], 784);
  return X;
}
