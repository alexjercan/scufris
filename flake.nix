{
  description = "A basic flake for my Scufris Project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }: (
    flake-utils.lib.eachDefaultSystem
    (system: let
      pkgs = import nixpkgs {
        inherit system;

        config = {
          allowUnfree = true;
        };
      };
      ollama-cuda = pkgs.ollama.override {acceleration = "cuda";};
    in {
      packages.default = pkgs.buildGoModule {
        pname = "scufris";
        version = "0.1";

        src = ./.;

        vendorHash = null;

        subPackages = ["cmd/scufris"];

        meta = {
          description = "Define your flatpak applications and permissions";
          homepage = "https://github.com/alexjercan/scufris";
          license = pkgs.lib.licenses.mit;
          maintainers = [];
          mainProgram = "scufris";
        };
      };

      devShells.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          go
          ollama-cuda
        ];

        buildInputs = [];
      };
    })
  );
}
