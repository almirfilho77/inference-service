package inference

import (
	"image/jpeg"
	"image"
	"image/color"
	"image/draw"
	"log"
	"os"
	"path/filepath"

	// Package image/jpeg is not used explicitly in the code below,
	// but is imported for its initialization side-effect, which allows
	// image.Decode to understand JPEG formatted images. Uncomment these
	// two lines to also understand GIF and PNG images:
	// _ "image/gif"
	// _ "image/png"
)

type Inference struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	Size int64  `json:"size"`
}

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

// TODO: separate the responsibilities of this package with maybe an image_util package
func RunInference(img image.Image, info Inference) string {
	// Modify the uploaded image to see if it works
	pink := color.RGBA{255, 0, 255, 255}
	imgBounds := img.Bounds()
	tempMaxX := 1930 - 1082
	tempMaxY := 1296 - 188
	tempImage := image.NewRGBA(image.Rect(0, 0, tempMaxX, tempMaxY))
	draw.Draw(tempImage, imgBounds, img, image.Point{1082, 188}, draw.Src)
	drawBoundingBox(tempImage, pink, image.Rect(tempMaxX/2, tempMaxY/2, 100, 100))

	Inferences = append(Inferences, info)
	log.Println("Add new inference")
	for _, inf := range Inferences {
		log.Println("Inferences")
		log.Printf("ID : %s", inf.ID)
		log.Printf("Name : %s", inf.Name)
		log.Printf("Size : %d", inf.Size)
	}
	inferenceMap[info.ID] = filepath.Join(".", "inferences", info.Name+".jpeg")
	writeToFile(tempImage, inferenceMap[info.ID])
	return inferenceMap[info.ID]
}

func drawBoundingBox(img draw.Image, color color.Color, rect image.Rectangle) {
	minX := rect.Min.X
	minY := rect.Min.Y
	maxX := rect.Max.X
	maxY := rect.Max.Y

	for i := minX; i < maxX; i++ {
		img.Set(i, minY, color)
		img.Set(i, maxY, color)
	}

	for j := minY; j < maxY; j++ {
		img.Set(minX, j, color)
		img.Set(maxX, j, color)
	}
}

// TODO: stablish the point where the path will be absolute (why from this point only??)
func writeToFile(img draw.Image, filePath string) {
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
	opt := jpeg.Options{Quality: 100}
	jpeg.Encode(outputFile, img, &opt)
}

func InitModel() error {
	inferenceMap = make(map[string]string)
	return nil
}
