"""``python -m gateway`` -- run the gateway with uvicorn."""
from __future__ import annotations

import os

import uvicorn

from inference_platform.logging_setup import configure_logging


def main() -> None:
    configure_logging("gateway")
    uvicorn.run(
        "gateway.app:app",
        host=os.environ.get("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.environ.get("GATEWAY_PORT", "8080")),
        workers=int(os.environ.get("GATEWAY_WORKERS", "1")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
