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
      buildGoApplication = pkgs.buildGoApplication;

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
          mainProgram = "scufris-service";
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
          mainProgram = "scufris-client";
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
      in {
        options.services.scufris = {
          enable = lib.mkEnableOption "The Scufris AI Assistant Service";
        };

        config = lib.mkIf cfg.enable {
          systemd.user.services.scufris = {
            Unit = {
              Description = "Scufris AI Assistant Service";
              After = ["network.target"];
            };

            Service = {
              ExecStart = "${scufris-service}/bin/scufris-service";
              Restart = "on-failure";
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
