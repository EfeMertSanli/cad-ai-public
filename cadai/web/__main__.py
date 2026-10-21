import sys

import uvicorn

from cadai import config

if __name__ == "__main__":
    if config.HOST not in ("127.0.0.1", "localhost") and not config.PASSWORD:
        sys.exit("Ağa açmak için şifre gerekli: CADAI_PASSWORD ayarlayın (start_remote.bat). / "
                 "A password is needed to open the site to the network: set CADAI_PASSWORD (start_remote.bat).")
    print(f"cad-ai: http://{config.HOST}:{config.PORT}" + (" (şifreli / password protected)" if config.PASSWORD else ""))
    uvicorn.run("cadai.web.app:app", host=config.HOST, port=config.PORT, log_level="warning")
