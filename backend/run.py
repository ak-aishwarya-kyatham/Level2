import uvicorn
from uvicorn import Config, Server

if __name__ == "__main__":
    config = Config(
        app="app.main:app",
        host="127.0.0.1",
        port=8000,
        timeout_notify=120,
        log_level="info"
    )
    server = Server(config=config)
    server.run()
