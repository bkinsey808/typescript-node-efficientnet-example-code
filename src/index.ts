import { promises as fs } from "fs";
import * as nodeFetch from "node-fetch";
import {
  EfficientNetCheckPointFactory,
  EfficientNetCheckPoint,
} from "node-efficientnet";

const imageFileNames = ["car.jpg", "panda.jpg", "fish.jpg", "gun.jpg"];
const imageDir = "./samples";
const localModelRootDirectory = "./models";
const imageDirRemoteUri =
  "https://raw.githubusercontent.com/ntedgi/node-EfficientNet/main/samples";

const download = async (imageFileName: string) => {
  const path = `${imageDir}/${imageFileName}`;
  try {
    // it's faster to check if the file already exist locally
    await fs.stat(path);
  } catch (err) {
    const response = await nodeFetch.default(
      `${imageDirRemoteUri}/${imageFileName}`
    );
    const buffer = await response.buffer();
    await fs.writeFile(path, buffer);
  }
};

const main = async () => {
  console.time("main");
  try {
    // it is not recommended to check if dir exist before mkdir, instead just run mkdir and don't catch the error
    try {
      await fs.mkdir(imageDir);
    } catch (e) {
      if (e.code !== "EEXIST") {
        throw e;
      }
    }

    let useLocalModels = true;
    try {
      await fs.stat(localModelRootDirectory);
    } catch (err) {
      if (err.code !== "ENOENT") {
        throw err;
      }
      useLocalModels = false;
    }

    const model = useLocalModels
      ? await EfficientNetCheckPointFactory.create(EfficientNetCheckPoint.B7, {
          localModelRootDirectory,
        })
      : await EfficientNetCheckPointFactory.create(EfficientNetCheckPoint.B7);

    const inferencePromises = imageFileNames.map(async (imageFileName) => {
      await download(imageFileName);
      return await model.inference(`${imageDir}/${imageFileName}`);
    });

    const inferencePromiseResults = await Promise.allSettled(inferencePromises);

    const inferences = inferencePromiseResults
      .filter((inference) => inference.status === "fulfilled")
      .map(
        (inference) => (inference as PromiseFulfilledResult<any>).value.result
      );

    console.log(inferences);
    console.timeEnd("main");
  } catch (e) {
    console.error(e);
  }
};
main();
