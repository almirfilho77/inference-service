package server

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"inference-service/inference"
)

func getInferences(c *gin.Context) {
	inferences := inference.GetInferences()
	c.IndentedJSON(http.StatusOK, inferences)
}

// TODO: implement logic to manipulate different image formats and base64 encoding
// TODO: implement timer to figure out the running time of this core function
// TODO: add header to get raw results (without non-maximum suppresion)
// TODO: add header to set probability threshold to ignore object
func postDetectObjects(c *gin.Context) {
	var newInferenceInfo inference.Inference
	newInferenceInfo.ID = uuid.NewString()

	imgFileHeader, err := c.FormFile("image")
	if err != nil {
		log.Fatal(err)
	}
	newInferenceInfo.Size = imgFileHeader.Size
	log.Printf("Image filename: %s", imgFileHeader.Filename)
	log.Printf("Image size: %d", imgFileHeader.Size)

	imgFile, err := imgFileHeader.Open()
	if err != nil {
		log.Fatal(err)
	}

	imgName, _ := c.GetPostForm("name")
	newInferenceInfo.Name = imgName
	log.Printf("Image name: %s", imgName)

	boundingBoxJson, err:= inference.RunObjectDetection(imgFile, newInferenceInfo)
	if err != nil {
		c.JSON(http.StatusUnsupportedMediaType, nil)
	} else {
		c.JSON(http.StatusOK, boundingBoxJson)
	}
}

func InitWithGin() {
	router := gin.Default()
	router.MaxMultipartMemory = 8 << 20 // 8 MiB :: Set a lower memory limit for multipart forms (default is 32 MiB)
	router.GET("/inference", getInferences)
	router.POST("/detect_objects", postDetectObjects)
	router.Run("localhost:8080")
}
