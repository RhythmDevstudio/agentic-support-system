"""Vercel serverless entrypoint.

Vercel's Python runtime looks for a FastAPI instance named `app` at a supported
entrypoint (app.py / index.py / server.py / main.py at the project root, or the
same names inside src/ or app/). This file just re-exports the real app so the
actual application code stays under app/api/main.py, unaffected by deployment
platform.
"""

from app.api.main import app

__all__ = ["app"]
