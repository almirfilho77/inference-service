package inference

import (
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/nfnt/resize" //Copyright (c) 2012, Jan Schlicht <jan.schlicht@gmail.com>
	ort "github.com/yalue/onnxruntime_go" //Copyright (c) 2023 Nathan Otterness
)

// IMPORTANT:
// PyTorch: starting from 'yolov8s.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400)

type Inference struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	Size int64  `json:"size"`
}

// Nice reference about JSON in Go: https://go.dev/blog/json
type BoundingBox struct {
	Center_x    int64   `json:"cx"`
	Center_y    int64   `json:"cy"`
	Width       int64   `json:"width"`
	Height      int64   `json:"height"`
	Probability float32 `json:"probability"`
	Class       string  `json:"class"`
}

const probabilityThreshold = float32(0.5)

// Bounding Box color definitions
var boundingBoxColorPink = color.RGBA{255, 0, 255, 255}

// Holds the informations to be serialized in the JSON
var Inferences []Inference

// Holds the {ID:path} for the inferences
var inferenceMap map[string]string

func GetInferences() []Inference {
	return Inferences
}

func GetInference(id string) string {
	return inferenceMap[id]
}

func RunObjectDetection(imgBuffer io.Reader, inferenceInfo Inference) []BoundingBox {
	// read the io.Reader
	inputImg, formatName, err := image.Decode(imgBuffer)
	if err != nil {
		log.Fatal(err)
	}

	// resize the input image to 640x640 (create this copy to avoid having to resize back at the end)
	resizedInputImage := resize.Resize(640, 640, inputImg, resize.Lanczos3)

	// create the input tensor (1,3,640,640)
	inputArray := convertImageToFloat32Array(resizedInputImage, 640, 640)

	// run inference
	outputData := getOutputFromModel(inputArray)

	// post process output
	originalWidth := int64(inputImg.Bounds().Size().X)
	originalHeight := int64(inputImg.Bounds().Size().Y)
	boundingBoxResults := processOutput(outputData, originalWidth, originalHeight)

	// draw bounding boxes
	// TODO: start a go coroutine to draw bounding boxes since the image is not going to be returned via http
	imgWithBBs := drawBoundingBox(inputImg, boundingBoxColorPink, boundingBoxResults)
	imgWithBBsPath := filepath.Join(".", "inferences", inferenceInfo.Name + "." + formatName)
	writeToFile(imgWithBBs, imgWithBBsPath)
	inferenceMap[inferenceInfo.ID] = imgWithBBsPath

	return boundingBoxResults
}

func convertImageToFloat32Array(inputImg image.Image, width int, height int) []float32 {
	var redChannel = []float32{}
	var greenChannel = []float32{}
	var blueChannel = []float32{}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := inputImg.At(x, y).RGBA()

			if x == 400 && y == 400 {
				log.Printf("r value : %d / after converted: %f", r, float32(r/257)/255.0)
			}

			redChannel = append(redChannel, float32(r/257)/255.0)
			greenChannel = append(greenChannel, float32(g/257)/255.0)
			blueChannel = append(blueChannel, float32(b/257)/255.0)
		}
	}
	resultingBuffer := append(redChannel, greenChannel...)
	resultingBuffer = append(resultingBuffer, blueChannel...)

	return resultingBuffer
}

func getOutputFromModel(inputArray []float32) []float32 {
	err := ort.InitializeEnvironment()
	if err != nil {
		ort.SetSharedLibraryPath(getSharedLibPath())
		err = ort.InitializeEnvironment()
		if err != nil {
			log.Fatal(err)
		}
	}
	defer ort.DestroyEnvironment()

	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewTensor(inputShape, inputArray)
	if err != nil {
		log.Fatal(err)
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatal(err)
	}
	defer outputTensor.Destroy()

	model, err := ort.NewAdvancedSession("./yolov8s.onnx",
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Destroy()

	err = model.Run()
	if err != nil {
		log.Fatal(err)
	}

	return outputTensor.GetData()
}

// TODO: Add non-maxima supression
func processOutput(inferenceResults []float32, width int64, height int64) []BoundingBox {
	resultingBoundingBoxes := []BoundingBox{}
	for bbIndex := 0; bbIndex < 8400; bbIndex++ {
		classIndex := 0
		highestProbability := float32(0.0)
		for col := 4; col < 84; col++ {
			classProbability := inferenceResults[8400*col+bbIndex]
			if classProbability > highestProbability {
				highestProbability = classProbability
				classIndex = col - 4
			}
		}

		if highestProbability < probabilityThreshold {
			continue
		}

		cx := int64(inferenceResults[bbIndex] * float32(width) / 640)
		cy := int64(inferenceResults[8400 + bbIndex] * float32(height) / 640)
		w := int64(inferenceResults[8400 * 2 + bbIndex] * float32(width) / 640)
		h := int64(inferenceResults[8400 * 3 + bbIndex] * float32(height) / 640)

		currentBoundingBox := BoundingBox{
			Center_x:    cx,
			Center_y:    cy,
			Width:       w,
			Height:      h,
			Probability: highestProbability,
			Class:       yolo_classes[classIndex],
		}
		currentBoundingBox.print()
		resultingBoundingBoxes = append(resultingBoundingBoxes, currentBoundingBox)
	}
	// filteredResults := supressNonMaximum(resultingBoundingBoxes)

	return resultingBoundingBoxes
}

// func supressNonMaximum(unfilteredBoundingBoxes []BoundingBox) []BoundingBox {
// 	filteredBoundingBoxes := []BoundingBox{}

// 	for _, bb := range unfilteredBoundingBoxes {
		
// 	}

// 	return filteredBoundingBoxes
// }

func (bb BoundingBox) print() {
	log.Println("\ncx: ", bb.Center_x, "\ncy: ", bb.Center_y, "\nw: ", bb.Width, "\nh: ", bb.Height, "\nprobability: ", bb.Probability, "\nclass: ", bb.Class)
}

func printBoundingBoxSlice(bbSlice []BoundingBox) {
	log.Println("Bounding box slice printing...")
	for _, bb := range bbSlice {
		bb.print()
	}
}

func drawBoundingBox(inputImg image.Image, color color.Color, boundingBoxes []BoundingBox) image.Image {
	// printBoundingBoxSlice(boundingBoxes)
	originalWidth := int64(inputImg.Bounds().Size().X)
	originalHeight := int64(inputImg.Bounds().Size().Y)
	resultImage := image.NewRGBA(image.Rect(0, 0, int(originalWidth), int(originalHeight)))
	draw.Draw(resultImage, inputImg.Bounds(), inputImg, image.Point{}, draw.Src)
	for _, bb := range boundingBoxes {
		minX := int(bb.Center_x - bb.Width/2)
		minY := int(bb.Center_y - bb.Height/2)
		maxX := int(bb.Center_x + bb.Width/2)
		maxY := int(bb.Center_y + bb.Height/2)
		
		for i := minX; i < maxX; i++ {
			resultImage.Set(i, minY, color)
			resultImage.Set(i, maxY, color)
		}
		
		for j := minY; j < maxY; j++ {
			resultImage.Set(minX, j, color)
			resultImage.Set(maxX, j, color)
		}
	}
	return resultImage
}

// TODO: stablish the point where the path will be absolute (why from this point only??)
func writeToFile(img image.Image, filePath string) {
	absFilePath, err := filepath.Abs(filePath)
	if err != nil {
		panic(err)
	}
	
	absDirPath := filepath.Dir(absFilePath)
	if _, err := os.Stat(absDirPath); os.IsNotExist(err) {
		err := os.MkdirAll(absDirPath, 0666)
		if err != nil {
			panic(err)
		}
	}
	
	outputFile, err := os.Create(absFilePath)
	if err != nil {
		panic(err)
	}
	defer outputFile.Close()
	
	// TODO: implement logic to manipulate different image formats and base64 encoding
	opt := jpeg.Options{Quality: 100}
	jpeg.Encode(outputFile, img, &opt)
}

// TODO: Get environment path variable pointing to the DLL and put this in the description
func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.so"
		}
		return "../third_party/onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}

// TODO: think if this is necessary and/or can be moved to somewhere else
func InitModel() error {
	inferenceMap = make(map[string]string)
	return nil
}

// Array of YOLOv8 class labels
var yolo_classes = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
