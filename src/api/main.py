from fastapi import FastAPI
from .routers.models import router as models_router


app = FastAPI(
    title="Trustworthy Model Registry",
    version="0.1.0",
    description="Baseline Delivery 1: CRUD, Ingest, Enumerate, Reset (+rate)",
)

@app.get("/")
def root():
    return {"message": "Trustworthy Model Registry API is running. Visit /docs for Swagger UI."}


app.include_router(models_router, prefix="", tags=["models"])

