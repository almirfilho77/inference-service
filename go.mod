module inference-service

go 1.21.5

replace inference-service/server => ./server

require (
	inference-service/inference v0.0.0-00010101000000-000000000000
	inference-service/server v0.0.0-00010101000000-000000000000
)

replace inference-service/inference => ./inference
