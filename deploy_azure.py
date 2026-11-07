"""
Deploy CarbIQ to Azure Container Apps (works on an Azure for Students subscription).

The image itself is built by GitHub Actions (.github/workflows/container-image.yml)
and published to ghcr.io; this script only tells Azure to run it.

Usage (from the project root, inside the venv):
    az login                                  # once
    python deploy_azure.py                    # first deploy, or redeploy the latest image
    python deploy_azure.py --secrets          # also copy API keys from .env
    python deploy_azure.py --dry-run          # print the az commands, run nothing

Sizing: 1 vCPU / 2 GiB (both models peak at ~0.9 GB). Scales to ZERO when idle,
so an unused app costs nothing; the first visit after idling takes ~1 minute.
Logs are not shipped to Log Analytics (avoids a per-GB ingestion charge);
live logs still work: az containerapp logs show -n carbiq -g carbiq-rg --follow

Secrets: SECRET_KEY is generated on first create. With --secrets, the API keys
from your local .env are stored as Container Apps SECRETS and exposed to the
app as env vars via secretref — they never appear in plain configuration.
"""
import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

SECRET_ENV = ("CLIMATIQ_API_KEY", "LLM_API_KEY", "YOUTUBE_API_KEY")
PLAIN_ENV = ("LLM_MODEL", "LLM_API_URL")

DRY_RUN = False


def az(*args, check=True, capture=False, secret=False):
    """Run an az CLI command. ``secret`` hides the arguments when printing."""
    cmd = [shutil.which("az") or "az", *args]
    shown = " ".join(args[:3]) + " ..." if secret else " ".join(args)
    print(f"> az {shown}")
    if DRY_RUN:
        return ""
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if check and res.returncode != 0:
        if capture:
            sys.stderr.write(res.stderr)
        sys.exit(f"az command failed: az {shown}")
    return (res.stdout or "").strip() if capture else res.returncode


def exists(*args) -> bool:
    if DRY_RUN:
        return False
    res = subprocess.run([shutil.which("az") or "az", *args],
                         capture_output=True, text=True)
    return res.returncode == 0


def read_env() -> dict:
    from dotenv import dotenv_values
    return {k: v for k, v in dotenv_values(os.path.join(ROOT, ".env")).items() if v}


def main() -> int:
    global DRY_RUN
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="carbiq")
    p.add_argument("--resource-group", default="carbiq-rg")
    p.add_argument("--location", default="southeastasia",
                   help="Azure region (default southeastasia = Singapore)")
    p.add_argument("--image", default="ghcr.io/k01j10w19/carbiq:latest")
    p.add_argument("--secrets", action="store_true", help="copy API keys from .env")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    DRY_RUN = a.dry_run

    if not DRY_RUN and not shutil.which("az"):
        sys.exit("Azure CLI not found. Install it, open a NEW terminal, then run `az login`.")

    name, rg, env_name = a.name, a.resource_group, f"{a.name}-env"

    az("account", "show", "--query", "name", "-o", "tsv")
    az("extension", "add", "--name", "containerapp", "--upgrade", "--yes")
    az("provider", "register", "--namespace", "Microsoft.App", "--wait")

    if exists("containerapp", "show", "-n", name, "-g", rg):
        # Same :latest tag -> a fresh revision suffix forces Azure to re-pull.
        az("containerapp", "update", "-n", name, "-g", rg, "--image", a.image,
           "--revision-suffix", time.strftime("r%Y%m%d%H%M%S"))
    else:
        az("group", "create", "-n", rg, "-l", a.location, "-o", "none")
        if not exists("containerapp", "env", "show", "-n", env_name, "-g", rg):
            az("containerapp", "env", "create", "-n", env_name, "-g", rg,
               "-l", a.location, "--logs-destination", "none", "-o", "none")
        az("containerapp", "create", "-n", name, "-g", rg,
           "--environment", env_name, "--image", a.image,
           "--target-port", "7860", "--ingress", "external",
           "--cpu", "1.0", "--memory", "2.0Gi",
           "--min-replicas", "0", "--max-replicas", "1",
           "--secrets", f"secret-key={secrets.token_hex(32)}",
           "--env-vars", "SECRET_KEY=secretref:secret-key",
           "-o", "none", secret=True)

    if a.secrets:
        env = read_env()
        found = [k for k in SECRET_ENV if k in env]
        if found:
            az("containerapp", "secret", "set", "-n", name, "-g", rg, "--secrets",
               *[f"{k.lower().replace('_', '-')}={env[k]}" for k in found],
               "-o", "none", secret=True)
        env_vars = [f"{k}=secretref:{k.lower().replace('_', '-')}" for k in found]
        env_vars += [f"{k}={env[k]}" for k in PLAIN_ENV if k in env]
        if env_vars:
            az("containerapp", "update", "-n", name, "-g", rg,
               "--set-env-vars", *env_vars, "-o", "none")
        print(f"  secrets set: {', '.join(found) or 'none found in .env'}")

    fqdn = az("containerapp", "show", "-n", name, "-g", rg, "--query",
              "properties.configuration.ingress.fqdn", "-o", "tsv", capture=True)
    print(f"\nApp URL: https://{fqdn or '<fqdn>'}")
    print("First load after idle takes ~1 minute (image pull + model load).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
