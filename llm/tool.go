package llm

import (
	"strings"

	"github.com/invopop/jsonschema"
)

type FunctionToolInfo struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  *jsonschema.Schema `json:"parameters"`
}

type ToolInfo struct {
	Type     string           `json:"type"`
	Function FunctionToolInfo `json:"function"`
}

func NewFunctionToolInfo(name string, description string, parameters any) FunctionToolInfo {
	// NOTE: Kind of hacky, but we need to extract the reference from the schema
	schema := jsonschema.Reflect(parameters)
	ref := strings.Split(schema.Ref, "/")[2]

	return FunctionToolInfo{
		Name:        name,
		Description: description,
		Parameters:  schema.Definitions[ref],
	}
}
