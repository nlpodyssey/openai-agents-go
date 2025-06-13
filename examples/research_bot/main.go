package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
)

func main() {
	fmt.Print("What would you like to research? ")
	_ = os.Stdout.Sync()

	line, _, err := bufio.NewReader(os.Stdin).ReadLine()
	if err != nil {
		panic(err)
	}
	query := string(line)

	err = NewResearchManager().Run(context.Background(), query)
	if err != nil {
		panic(err)
	}
}
