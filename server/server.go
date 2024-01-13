package server

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

func Init(addr string, handler http.Handler, read_timeout uint, write_timeout uint, max_header_bytes int){
	log.SetPrefix("inference-service/server")
	s := &http.Server{
		Addr:           addr,
		Handler:        handler,
		ReadTimeout:    time.Duration(read_timeout) * time.Second,
		WriteTimeout:   time.Duration(write_timeout) * time.Second,
		MaxHeaderBytes: max_header_bytes,
	}
	http.HandleFunc("/hello", helloHandler)
	log.Fatal(s.ListenAndServe())
}
