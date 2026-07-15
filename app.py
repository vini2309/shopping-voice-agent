import os

import uvicorn

from backend.app.main import app


def _register_zerogpu_marker() -> None:
    try:
        import spaces  # type: ignore
    except Exception:
        return

    @spaces.GPU
    def _zerogpu_marker() -> str:
        return "ready"

    globals()["_zerogpu_marker"] = _zerogpu_marker


_register_zerogpu_marker()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
