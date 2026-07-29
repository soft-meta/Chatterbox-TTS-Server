from __future__ import annotations

import logging

import uvicorn

from config import load_config

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    server = load_config()["server"]
    uvicorn.run(
        "server:app",
        host=server["host"],
        port=int(server["port"]),
        reload=False,
        access_log=True,
    )
