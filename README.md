# Machine Learning inference service

:bangbang: **THIS IS A WORK IN PROGRESS** :blush:

A Golang-based service that serves machine learning model for inference. Access a RESTful endpoint to run an inference on an instance of a YOLOv8 model.

## Requirements

Because of [onnxruntime bindings for Go](https://github.com/yalue/onnxruntime_go), if you are on Windows, you will need to be using a version of Go with cgo enabled---meaning that you'll need to have gcc available on your PATH. For this I recommend using [TDM-GCC-64](https://github.com/jmeubank/tdm-gcc/releases), since it was the only one that worked out for me.

As of right now, you need to put the lib files for onnxruntime in a folder that respects the following path: "../third_party/onnxruntime/onnxruntime.dll", because that is how I implemented the loading of the dynamic library so far. In the future, I'll get it from the PATH variable.

## APIs

### GET

- /inference : get a json with a list of previously ran inferences

### POST

- /detect_objects : send an image and get a json with the descriptions of the bounding boxes with the detected objects inside

        Request body "key":value to send:

        * "image" : jpeg or png image file
        * "name" : string with the name of the image (human friendly name or alias)

## Contact

To contact me, write me an email at <almir.filho77@gmail.com>.
