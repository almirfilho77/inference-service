package inference

import (
	"encoding/base64"
	"fmt"
	"image"
	"log"
	"os"
	"strings"

	// Package image/jpeg is not used explicitly in the code below,
	// but is imported for its initialization side-effect, which allows
	// image.Decode to understand JPEG formatted images. Uncomment these
	// two lines to also understand GIF and PNG images:
	// _ "image/gif"
	// _ "image/png"
	_ "image/jpeg"
)

type Inference struct {
	ID		string	`json:"id"`
	Name 	string 	`json:"name"`
	Size 	int64 	`json:"size"`
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

//func RunInference(img image.Image, info Inference) image.Image
func RunInference(info Inference) {
	Inferences = append(Inferences, info)
	log.Println("Add new inference")
	for _, inf := range Inferences {
		log.Println("Inferences")
		log.Printf("ID : %s", inf.ID)
		log.Printf("Name : %s", inf.Name)
		log.Printf("Size : %d", inf.Size)
	}
}


func PrintHistogram(img image.Image) {
	// Calculate a 16-bin histogram for m's red, green, blue and alpha components.
	//
	// An image's bounds do not necessarily start at (0, 0), so the two loops start
	// at bounds.Min.Y and bounds.Min.X. Looping over Y first and X second is more
	// likely to result in better memory access patterns than X first and Y second.
	bounds := img.Bounds()
	var histogram [16][4]int
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			// A color's RGBA method returns values in the range [0, 65535].
			// Shifting by 12 reduces this to the range [0, 15].
			histogram[r>>12][0]++
			histogram[g>>12][1]++
			histogram[b>>12][2]++
			histogram[a>>12][3]++
		}
	}
	fmt.Printf("%-14s %6s %6s %6s %6s\n", "bin", "red", "green", "blue", "alpha")
	for i, x := range histogram {
		fmt.Printf("0x%04x-0x%04x: %6d %6d %6d %6d\n", i<<12, (i+1)<<12-1, x[0], x[1], x[2], x[3])
	}
}

func LoadImage(filepath string) {
	// Decode the JPEG data. If reading from file, create a reader
	reader, err := os.Open(filepath)
	if err != nil {
	    log.Fatal(err)
	}
	defer reader.Close()
	img, formatName, err := image.Decode(reader)
	log.Printf("The image format is [%s]", formatName)
	PrintHistogram(img)
}

func LoadImageFromData(data string) (image.Image, string) {
	reader := base64.NewDecoder(base64.StdEncoding, strings.NewReader(data))
	img, formatName, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("The image format is [%s]", formatName)
	PrintHistogram(img)
	return img,formatName
}