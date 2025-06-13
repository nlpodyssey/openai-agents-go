package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
)

func main() {
	fmt.Print("Enter a financial research query: ")
	_ = os.Stdout.Sync()

	line, _, err := bufio.NewReader(os.Stdin).ReadLine()
	if err != nil {
		panic(err)
	}
	query := string(line)

	err = NewFinancialResearchManager().Run(context.Background(), query)
	if err != nil {
		panic(err)
	}
}
