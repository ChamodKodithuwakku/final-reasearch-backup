'use strict'
import path from 'path';
import _ from "lodash";
import * as tf from '@tensorflow/tfjs-node';
import cv from 'opencv4nodejs'
//const util = require('./util')
import * as util from './util.js';

import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);


const FACE_MODEL_PATH = path.join(__dirname, '../models/haarcascade_frontalface_default.xml')
const faceModel = new cv.CascadeClassifier(FACE_MODEL_PATH)

export async function getFaces (image) {
  const facesResult = await faceModel.detectMultiScaleAsync(image)

  const faces = facesResult.objects.map(function (face, i) {
    if (facesResult.numDetections[i] < 10) return null

    return face
  })

  return faces.filter(function (face) { return face })
}

export async function preprocessToTensor (faceImage, targetSize) {
  faceImage = await faceImage.resizeAsync(targetSize[0], targetSize[1])
  faceImage = await faceImage.bgrToGrayAsync()

  let tensor = tf.tensor3d(_.flattenDeep(faceImage.getDataAsArray()), [64, 64, 1])

  tensor = tensor.asType('float32')
  tensor = tensor.div(255.0)
  tensor = tensor.sub(0.5)
  tensor = tensor.mul(2.0)
  tensor = tensor.reshape([1, 64, 64, 1])

  return tensor
}

export async function inferEmotion (tensor, model) {
  const result = await model.predict(tensor)

  console.log("in results" + result)
  return util.getEmotionLabel(result)

}
