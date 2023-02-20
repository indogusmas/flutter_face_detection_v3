import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as imglib;
import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

Future<CameraDescription> getCamera(CameraLensDirection dir) async {
  //There is CameraLensDirection
  //CameraLensDirection.front,
  //CameraLensDirection.back
  //You can custom According to Requirement
  //In sample i use CameraLensDirecrion.front
  return await availableCameras().then(
        (List<CameraDescription> cameras) => cameras.firstWhere(
          (CameraDescription camera) => camera.lensDirection == dir,
    ),
  );
}

Future<List<Face>> detect(CameraImage image,
    FaceDetector faceDetector,
    InputImageRotation rotation) async {
  final WriteBuffer allBytes = WriteBuffer();
  for (Plane plane in image.planes) {
    allBytes.putUint8List(plane.bytes);
  }
  final bytes = allBytes.done().buffer.asUint8List();
  final Size imageSize = Size(image.width.toDouble(), image.height.toDouble());
  print("Size image $imageSize" );

  final inputImageData = InputImageData(
    size: imageSize,
    imageRotation: rotation,
    inputImageFormat: InputImageFormat.yuv420,
    planeData: image.planes.map(
          (Plane plane) {
        return InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        );
      },
    ).toList(),
  );
  InputImage _firebaseVisionImage = InputImage.fromBytes(
    bytes: bytes,
    inputImageData: inputImageData,
  );
  dynamic listFace =await faceDetector.processImage(_firebaseVisionImage);
  return  listFace;
}

InputImageRotation rotationIntToImageRotation(int rotation) {
  switch (rotation) {
    case 90:
      return InputImageRotation.rotation90deg;
    case 180:
      return InputImageRotation.rotation180deg;
    case 270:
      return InputImageRotation.rotation270deg;
    default:
      return InputImageRotation.rotation0deg;
  }
}

Float32List imageToByteListFloat32(
    imglib.Image image, int inputSize, double mean, double std) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;
  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      buffer[pixelIndex++] = (imglib.getRed(pixel) - mean) / std;
      buffer[pixelIndex++] = (imglib.getGreen(pixel) - mean) / std;
      buffer[pixelIndex++] = (imglib.getBlue(pixel) - mean) / std;
    }
  }
  return convertedBytes.buffer.asFloat32List();
}

double euclideanDistance(List e1, List e2) {
  double sum = 0.0;
  for (int i = 0; i < e1.length; i++) {
    sum += pow((e1[i] - e2[i]), 2);
  }
  return sqrt(sum);
}