import os, io, time, shutil
import requests
import streamlit as st

# gdown là lựa chọn ưu tiên; nếu thiếu, ta fallback sang requests
try:
    import gdown
    HAS_GDOWN = True
except Exception:
    HAS_GDOWN = False

def _drive_id_to_url(file_id: str) -> str:
    # link đơn giản; cần file để "Anyone with the link - Viewer"
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def safe_gdown(file_id: str, out_path: str, tries: int = 3) -> str:
    """
    Tải file Google Drive về out_path.
    Ưu tiên gdown (nếu có), nếu thất bại thì fallback sang requests.
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    last_err = None

    # 1) gdown (nếu sẵn có)
    if HAS_GDOWN:
        for _ in range(tries):
            try:
                gdown.download(id=file_id, output=out_path, quiet=False)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return out_path
            except Exception as e:
                last_err = e
                time.sleep(1.0)

    # 2) fallback: requests (đơn giản; có thể không vượt qua bước confirm của file rất lớn)
    try:
        url = _drive_id_to_url(file_id)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        last_err = e

    st.error(
        "Không tải được file từ Google Drive. "
        "Hãy chắc chắn file đã bật **Anyone with the link – Viewer**."
    )
    raise RuntimeError(f"Download failed for {file_id}: {last_err}")
