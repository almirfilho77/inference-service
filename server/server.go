package server

import (
	"image"
	_ "image/jpeg"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"inference-service/inference"
)

func getInferences(c *gin.Context) {
	inferences := inference.GetInferences()
	c.IndentedJSON(http.StatusOK, inferences)
}

// TODO: implement logic to manipulate different image formats and base64 encoding
func postInference(c *gin.Context) {
	var newInference inference.Inference
	img, err := c.FormFile("image")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Image filename: %s", img.Filename)
	log.Printf("Image size: %d", img.Size)

	imgFile, err := img.Open()
	if err != nil {
		log.Fatal(err)
	}
	forInferenceImage, imageFormat, err := image.Decode(imgFile)
	log.Printf("Image format: %s", imageFormat)

	name, _ := c.GetPostForm("name")
	log.Printf("Image name: %s", name)

	newInference.ID = uuid.NewString()
	newInference.Name = name
	newInference.Size = img.Size

	inferencedFilepath := inference.RunInference(forInferenceImage, newInference)
	outputImgFile, err := os.Open(inferencedFilepath)
	if err != nil {
		panic(err)
	}
	defer outputImgFile.Close()

	outputImg, err := io.ReadAll(outputImgFile)
	if err != nil {
		panic(err)
	}

	c.Writer.Header().Set("Content-Type", "image/jpeg")
	c.Writer.Write(outputImg)
}

func InitWithGin() {
	router := gin.Default()
	// Set a lower memory limit for multipart forms (default is 32 MiB)
	router.MaxMultipartMemory = 8 << 20 // 8 MiB
	router.GET("/inference", getInferences)
	router.POST("/inference", postInference)
	router.Run("localhost:8080")
}
