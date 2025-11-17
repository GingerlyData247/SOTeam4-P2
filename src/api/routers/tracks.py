from fastapi import APIRouter

router = APIRouter()

@router.get("/tracks")
def get_tracks():
    planned_tracks = ["Performance track"]  # casing is very important
    return {"plannedTracks": planned_tracks}
