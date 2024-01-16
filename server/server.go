package server

import (
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"inference-service/inference"
	"log"
	"net/http"
)

func getInferences(c *gin.Context) {
	inferences := inference.GetInferences()
	c.IndentedJSON(http.StatusOK, inferences)
}

func postInference(c *gin.Context) {
	var newInference inference.Inference
	img, err := c.FormFile("image")
	if err != nil {
		log.Fatal(err)
	}
	name, _ := c.GetPostForm("name")

	log.Printf("Image name: %s", name)
	log.Printf("Image filename: %s", img.Filename)
	log.Printf("Image size: %d", img.Size)
	
	newInference.ID = uuid.NewString()
	newInference.Name = name
	newInference.Size = img.Size

	inference.RunInference(newInference)
}

func InitWithGin() {
	router := gin.Default()
	// Set a lower memory limit for multipart forms (default is 32 MiB)
	router.MaxMultipartMemory = 8 << 20  // 8 MiB
	router.GET("/inference", getInferences)
	router.POST("/inference", postInference)
	router.Run("localhost:8080")
}