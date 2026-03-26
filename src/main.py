from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(
    title="MVM² — Multi-LLM Math Verification System",
    description="Verifies math solutions using consensus across multiple AI models",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "MVM² Math Verification System",
        "version": "0.1.0",
        "docs": "/docs",
    }
