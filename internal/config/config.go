package config

import (
	"github.com/ilyakaznacheev/cleanenv"
)

// Somehow make these variables so I can set them from nix
type Config struct {
	ConfigPath string `env:"CONFIG_PATH" env-default:"config.yaml"`
	Database   struct {
		Host     string `yaml:"host"`
		Port     int    `yaml:"port"`
		User     string `yaml:"user"`
		Password string `yaml:"password"`
		Database string `yaml:"database"`
		Insecure bool   `yaml:"insecure"`
	} `yaml:"database"`
	Ollama struct {
		Url string `yaml:"url"`
	} `yaml:"ollama"`
	ImageGen struct {
		Url string `yaml:"url"`
	} `yaml:"imagegen"`
	EmbeddingModel string `yaml:"embedding_model"`
	SocketPath     string `yaml:"socket_path"`
}

func LoadConfig() (cfg Config, err error) {
	err = cleanenv.ReadEnv(&cfg)
	if err != nil {
		return
	}

	err = cleanenv.ReadConfig(cfg.ConfigPath, &cfg)
	if err != nil {
		return
	}

	return
}

type ClientConfig struct {
	ConfigPath string `env:"CONFIG_PATH" env-default:"config.yaml"`
	SocketPath string `yaml:"socket_path"`
}

func LoadClientConfig() (cfg ClientConfig, err error) {
	err = cleanenv.ReadEnv(&cfg)
	if err != nil {
		return
	}

	err = cleanenv.ReadConfig(cfg.ConfigPath, &cfg)
	if err != nil {
		return
	}

	return
}
