package main

import (
	"log"
	"net/http"
	"os"
)

func main() {
    // Open a log file
    file, err := os.OpenFile("server.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    // Set up logging to both file and terminal
    log.SetOutput(file)
    log.SetFlags(log.LstdFlags | log.Lshortfile)

    // Serve the index.html file and other static files
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        log.Printf("Received request: %s %s from %s\n", r.Method, r.URL.Path, r.RemoteAddr)
        if r.URL.Path == "/" {
            http.ServeFile(w, r, "index.html")
        } else {
            http.ServeFile(w, r, "."+r.URL.Path)
        }
    })

    log.Println("Starting server on :8080")
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatal(err)
    }
}
