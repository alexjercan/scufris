package imagegen

type GenerateRequest struct {
	Prompt string `json:"prompt"`
}

func NewGenerateRequest(prompt string) GenerateRequest {
	return GenerateRequest{
		Prompt: prompt,
	}
}
