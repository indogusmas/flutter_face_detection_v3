import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_face_detection_v3/FacePainter.dart';
import 'package:flutter_face_detection_v3/Utils.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as imglib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  late CameraController _cameraController;
  bool isReadyCamera = false;

  bool _isDetecting = false;
  bool _faceFound = false;
  dynamic _scanResults;
  var interpreter;
  dynamic data = {};
  double threshold = 1.0;
  late List e1;

  //Change to Map
  Map<String, Face> _mapFace = Map();
  List<Face> listFace = [];

  @override
  void initState() {
    super.initState();
    initCamera();
  }

  @override
  void dispose() {
    super.dispose();
    if(_cameraController !=  null) _cameraController.dispose();
  }

  FaceDetector _getFaceDetector() {
    final faceDetector = FaceDetector(
        options: FaceDetectorOptions(
            performanceMode: FaceDetectorMode.fast,
            enableContours: true,
            enableLandmarks: true,
            enableClassification: true
        )
    );
    return faceDetector;
  }

  Future loadModel() async {
    try {
      final gpuDelegateV2 = tfl.GpuDelegateV2(
          options: tfl.GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,
            inferencePreference: tfl.TfLiteGpuInferenceUsage.fastSingleAnswer,
            inferencePriority1: tfl.TfLiteGpuInferencePriority.minLatency,
            inferencePriority2: tfl.TfLiteGpuInferencePriority.auto,
            inferencePriority3: tfl.TfLiteGpuInferencePriority.auto,
          ));
      var interpreterOptions = tfl.InterpreterOptions()
        ..addDelegate(gpuDelegateV2);
      interpreter = await tfl.Interpreter.fromAsset('mobilefacenet.tflite',
          options: interpreterOptions);
    } on Exception {
      print('Failed to load model.');
    }
  }

  Size getImageSize() {
    assert(_cameraController != null, 'Camera controller not initialized');
    assert(
    _cameraController!.value.previewSize != null, 'Preview size is null');
    return Size(
      _cameraController!.value.previewSize!.height,
      _cameraController!.value.previewSize!.width,
    );
  }

  String _recog(imglib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, null, growable: false).reshape([1, 192]);
    interpreter.run(input, output);
    output = output.reshape([192]);
    e1 = List.from(output);
    return compare(e1).toUpperCase();
  }

  String compare(List currEmb) {
    if (data.length == 0) return "No Face saved";
    double minDist = 999;
    double currDist = 0.0;
    String predRes = "NOT RECOGNIZED";
    for (String label in data.keys) {
      currDist = euclideanDistance(data[label], currEmb);
      if (currDist <= threshold && currDist < minDist) {
        minDist = currDist;
        predRes = label;
      }
    }
    print(minDist.toString() + " " + predRes);
    return predRes;
  }

  Future initCamera() async {
    await loadModel();
    //Create a Camera Controller
    CameraDescription description = await getCamera(CameraLensDirection.front);
    InputImageRotation rotation = rotationIntToImageRotation(
      description.sensorOrientation,
    );
    _cameraController = CameraController(description, ResolutionPreset.low);
    try {
      await _cameraController.initialize().then((_) {
        if (!mounted) return;
        setState(() {
          isReadyCamera = true;
        });
      });
      if(_cameraController != null){
        _cameraController.startImageStream((image) {
          if(_cameraController != null){
            if(_isDetecting) return;
              _isDetecting = true;
            String res;
           // dynamic finalResult;
            Map<String, Face> mapFace = Map();
            detect(image,_getFaceDetector(),rotation).then(
                  (List<Face> result) async {
                if (result.length == 0)
                  _faceFound = false;
                else
                  _faceFound = true;
                print("FirebaseML listface 1 ${result}");
                Face _face;
                List<Face> _faceList = [];
                imglib.Image convertedImage =
                _convertCameraImage(image, CameraLensDirection.front);
                for (_face in result) {
                  double x, y, w, h;
                  x = (_face.boundingBox.left - 10);
                  y = (_face.boundingBox.top - 10);
                  w = (_face.boundingBox.width + 10);
                  h = (_face.boundingBox.height + 10);
                  imglib.Image croppedImage = imglib.copyCrop(
                      convertedImage, x.round(), y.round(), w.round(), h.round());
                  croppedImage = imglib.copyResizeCropSquare(croppedImage, 112);
                  // int startTime = new DateTime.now().millisecondsSinceEpoch;
                  res = _recog(croppedImage);
                  // int endTime = new DateTime.now().millisecondsSinceEpoch;
                  // print("Inference took ${endTime - startTime}ms");
                  // finalResult.add(res,_face);
                  // finalResult = "indo";
                  _faceList.add(_face);
                }
                setState(() {
                  listFace = _faceList;
                  //_scanResults = finalResult;
                  print("CameraPage Set State res " + listFace.length.toString() );
                });
                _isDetecting = false;
              },
            ).catchError((e) {
              print("FirebaseML listface 3 $e");
              setState(() {
                _isDetecting = false;
              });
            });
          }else{
            print("Null stream");
          }
        });
      }
    } on CameraException catch (e) {
      debugPrint("Camera error $e");
    }
  }


  imglib.Image _convertCameraImage(
      CameraImage image, CameraLensDirection _dir) {
    int width = image.width;
    int height = image.height;
    // imglib -> Image package from https://pub.dartlang.org/packages/image
    var img = imglib.Image(width, height); // Create Image buffer
    const int hexFF = 0xFF000000;
    final int uvyButtonStride = image.planes[1].bytesPerRow;
    final int? uvPixelStride = image.planes[1].bytesPerPixel;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        final int uvIndex =
            uvPixelStride! * (x / 2).floor() + uvyButtonStride * (y / 2).floor();
        final int index = y * width + x;
        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        // Calculate pixel color
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        // color: 0x FF  FF  FF  FF
        //           A   B   G   R
        img.data[index] = hexFF | (b << 16) | (g << 8) | r;
      }
    }
    var img1 = (_dir == CameraLensDirection.front)
        ? imglib.copyRotate(img, -90)
        : imglib.copyRotate(img, 90);
    return img1;
  }

  Widget _buildResults() {
    const Text noResultsText = const Text('');
    if (listFace.length == 0) {
      return noResultsText;
    }else{
      //final painter = FaceDetectorPainter(getImageSize(),_scanResults);
     //return Container(child: ElevatedButton(onPressed: () {}, child: const Text("Face Detection")));
      Face face = listFace[0];
      print("CameraPage Set State res " + face.toString());
    return CustomPaint(
        painter: FacePainter(
          face: face,
          imageSize: getImageSize()
        ),
      );
     // print("CameraPage customPaint ${custom}");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
          child: Stack(children: [
            (isReadyCamera && _cameraController.value.isInitialized)
                ? Stack(
              fit: StackFit.expand,
              children: <Widget>[
                CameraPreview(_cameraController),
                _buildResults(),
              ],
            )
                : Container(
                color: Colors.black,
                child: const Center(child: CircularProgressIndicator())),
          ])
      ),
    );
  }
}
