{
  description = "A basic flake for my Scufris Project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    gomod2nix.url = "github:nix-community/gomod2nix";
    gomod2nix.inputs.nixpkgs.follows = "nixpkgs";
    gomod2nix.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    gomod2nix,
  }: (
    flake-utils.lib.eachDefaultSystem
    (system: let
      pkgs = import nixpkgs {
        inherit system;

        overlays = [
          (import "${gomod2nix}/overlay.nix")
        ];

        config = {
          allowUnfree = true;
        };
      };

      goEnv = pkgs.mkGoEnv {pwd = ./.;};
      buildGoApplication = pkgs.gomod2nix.buildGoApplication;

      scufris-service = buildGoApplication {
        pname = "scufris-service";
        version = "0.1.0";
        src = ./.;
        modules = ./gomod2nix.toml;

        subPackages = ["cmd/server"];

        meta = {
          description = "The Scufris AI Assistant Service";
          homepage = "https://github.com/alexjercan/scufris";
          license = pkgs.lib.licenses.mit;
          maintainers = [];
          mainProgram = "server";
        };
      };

      scufris-client = buildGoApplication {
        pname = "scufris-client";
        version = "0.1.0";
        src = ./.;
        modules = ./gomod2nix.toml;

        subPackages = ["cmd/client"];

        meta = {
          description = "The Scufris AI Assistant Client";
          homepage = "https://github.com/alexjercan/scufris";
          license = pkgs.lib.licenses.mit;
          maintainers = [];
          mainProgram = "client";
        };
      };

      scufris = buildGoApplication {
        pname = "scufris";
        version = "0.1.0";
        src = ./.;
        modules = ./gomod2nix.toml;

        subPackages = ["cmd/scufris"];

        meta = {
          description = "The Scufris AI Assistant Command Line Interface";
          homepage = "https://github.com/alexjercan/scufris";
          license = pkgs.lib.licenses.mit;
          maintainers = [];
          mainProgram = "scufris";
        };
      };
    in {
      packages = {
        inherit scufris-service scufris-client scufris;
        default = scufris;
      };

      apps = {
        default = flake-utils.lib.mkApp {
          drv = scufris;
        };
        service = flake-utils.lib.mkApp {
          drv = scufris-service;
        };
        client = flake-utils.lib.mkApp {
          drv = scufris-client;
        };
      };

      homeManagerModules.default = {
        config,
        lib,
        pkgs,
        ...
      }: let
        cfg = config.services.scufris;
        config_file = ".config/scufris/config.yaml";
      in {
        options.services.scufris = {
          enable = lib.mkEnableOption "The Scufris AI Assistant Service";

          database = {
            host = lib.mkOption {
              type = lib.types.str;
              default = "localhost";
              description = "Database host";
            };
            port = lib.mkOption {
              type = lib.types.int;
              default = 5432;
              description = "Database port";
            };
            user = lib.mkOption {
              type = lib.types.str;
              default = "scufris";
              description = "Database user";
            };
            password = lib.mkOption {
              type = lib.types.str;
              default = "scufris";
              description = "Database password";
            };
            database = lib.mkOption {
              type = lib.types.str;
              default = "scufris";
              description = "Database name";
            };
            insecure = lib.mkOption {
              type = lib.types.bool;
              default = true;
              description = "Whether to allow insecure connections to the database";
            };
          };
          ollama = {
            url = lib.mkOption {
              type = lib.types.str;
              default = "http://localhost:11434";
              description = "Ollama API URL";
            };
          };
          imagegen = {
            url = lib.mkOption {
              type = lib.types.str;
              default = "http://localhost:11435";
              description = "Image generation API URL";
            };
          };
          embeddingModel = lib.mkOption {
            type = lib.types.str;
            default = "nomic-embed-text";
            description = "Embedding model to use";
          };
          socketPath = lib.mkOption {
            type = lib.types.str;
            default = "/tmp/scufris.sock";
            description = "Socket path for the service";
          };
        };

        config = lib.mkIf cfg.enable {
          home.file."${config_file}".text =
            pkgs.lib.generators.toYAML {
            } {
              database = {
                host = cfg.database.host;
                port = cfg.database.port;
                user = cfg.database.user;
                password = cfg.database.password;
                database = cfg.database.database;
                insecure = cfg.database.insecure;
              };
              ollama = {
                url = cfg.ollama.url;
              };
              imagegen = {
                url = cfg.imagegen.url;
              };
              embedding_model = cfg.embeddingModel;
              socket_path = cfg.socketPath;
            };

          systemd.user.services.scufris = {
            Unit = {
              Description = "Scufris AI Assistant Service";
              After = ["network.target"];
            };

            Service = {
              ExecStart = "${scufris-service}/bin/scufris-service";
              Restart = "on-failure";

              Environment = [ "CONFIG_PATH=%h/${config_file}" ];
            };

            Install = {
              WantedBy = ["default.target"];
            };
          };

          home.packages = [scufris-client];
        };
      };

      devShells.default = pkgs.mkShell {
        nativeBuildInputs = [
          goEnv
          pkgs.gomod2nix
        ];

        buildInputs = [];
      };
    })
  );
}
