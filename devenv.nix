{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
let
  pkgs-unstable = import inputs.nixpkgs-unstable {
    system = pkgs.stdenv.system;
    config = {
      allowUnfree = true;
    };
  };
in
{
  imports = [
    inputs.githooks.modules.default
  ];

  custom.git-hooks = {
    jupyter-notebook.enable = true;
  };

  devcontainer.enable = true;

  # https://devenv.sh/packages/
  packages = with pkgs-unstable; [ claude-code ];

  # https://devenv.sh/languages/
  languages = {
    nix.enable = true;
    python = {
      enable = true;
      venv.enable = true;
      uv = {
        enable = true;
        sync.enable = true;
      };
    };
  };

  # https://devenv.sh/git-hooks/
  git-hooks = {
    hooks = {
      ruff.enable = true;
      ruff-format.enable = true;
    };
    package = pkgs.prek;
  };

  # See full reference at https://devenv.sh/reference/options/
}
