package main

import (
	"context"
	"net/http"
	"strconv"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type Input struct {
	Name     string `json:"name" jsonschema:"user name"`
	LastName string `json:"lastName" jsonschema:"user last name"`
	Grade    string `json:"grade,omitempty" jsonschema:"grade"`
}

type Output struct {
	Message string `json:"message" jsonschema:"message"`
}

func ComplexTool(ctx context.Context, req *mcp.CallToolRequest, input Input) (
	*mcp.CallToolResult,
	Output,
	error,
) {
	grade := []rune(input.Grade)
	if len(grade) == 0 {
		return nil, Output{
			Message: "SUCCESS: 101",
		}, nil
	}
	return nil, Output{
		Message: "Failed",
	}, nil
}

func runServer(port int) {
	server := mcp.NewServer(&mcp.Implementation{Name: "Some Server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "get_secret_code", Description: "Tool for getting secret code"}, ComplexTool)

	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server {
		return server
	}, nil)

	http.ListenAndServe(":"+strconv.Itoa(port), handler)
}
