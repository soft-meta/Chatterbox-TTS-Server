from __future__ import annotations

import uvicorn
from config import load_config

if __name__ == "__main__":
    config = load_config()["server"]
    uvicorn.run("server:app", host=config["host"], port=int(config["port"]), reload=False)
