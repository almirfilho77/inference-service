package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

func helloHandler(w http.ResponseWriter, req *http.Request) {
	log.Printf("Host [%s]", req.Host)
	log.Printf("Client [%s]", req.RemoteAddr)
	log.Printf("Length of request [%d]", req.ContentLength)
	nBytes, err := fmt.Fprintf(w, "hello\n")
	if err != nil {
		log.Println(err)
	}
	log.Printf("Bytes written [%d]", nBytes)
}

func main() {
	// Init logger
	log.SetPrefix("inference-service: ")
	log.SetFlags(15)

	// Load YOLO model

	// Init server
	s := &http.Server{
		Addr:           ":8080",
		Handler:        nil,
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}

	// Listen to requests from RESTful endpoint
	http.HandleFunc("/hello", helloHandler)
	log.Fatal(s.ListenAndServe())

	// Process request and send response back

}
