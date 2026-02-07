from fastapi import HTTPException, Request

import db


async def require_api_key(request: Request):
    auth_header = request.headers.get("authorization") or ""
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="invalid_key")
    api_key = auth_header.replace("Bearer ", "", 1).strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="invalid_key")
    user = db.get_user_by_key(api_key)
    if not user or not int(user["is_active"]):
        raise HTTPException(status_code=401, detail="invalid_key")
    return user
