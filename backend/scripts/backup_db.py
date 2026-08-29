import sqlite3
import logging
import pickle
from pathlib import Path
from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

#local
SOURCE_DB = Path(r"D:\SLAW\LAW---chatbot\backend\data\slaw_gemini.db")  # file database gốc
LOCAL_BACKUP_DIR = Path(r"D:\SLAW\LAW---chatbot\backend\backup\local")
LOCAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

#oauth (thay cho service account)
CLIENT_SECRET_FILE = Path(r"D:\SLAW\LAW---chatbot\backend\client_secret.json")
TOKEN_FILE = Path(r"D:\SLAW\LAW---chatbot\backend\token.pickle")
DRIVE_FOLDER_ID = "1Xr06V2Ja3BO_jH8jgEdCo_SP6PiakWEG"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

MAX_LOCAL_BACKUPS = 7  # giu 7 ban local gan nhat (backup hang ngay)
MAX_DRIVE_BACKUPS = 7  # giu 7 ban tren Drive gan nhat, dong bo voi local

LOG_FILE = Path(r"D:\SLAW\LAW---chatbot\backend\backup\backup.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("backup")


#local backup
def backup_to_local() -> Path:
    if not SOURCE_DB.exists():
        raise FileNotFoundError(f"Db nguon khong ton tai: {SOURCE_DB}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = LOCAL_BACKUP_DIR / f"state_{timestamp}.db"
    conn = sqlite3.connect(SOURCE_DB, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000;")
    try:
        conn.execute("VACUUM INTO ?", (str(backup_path),))
    finally:
        conn.close()
    return backup_path


def verify(backup_path: Path) -> bool:
    conn = sqlite3.connect(backup_path)
    try:
        result = conn.execute("PRAGMA integrity_check;").fetchone()  # tim loi hoac ban ghi bi thieu
        return result[0] == "ok"
    finally:
        conn.close()


#oauth 
def get_credentials() -> Credentials:
    if not CLIENT_SECRET_FILE.exists():
        raise FileNotFoundError(f"Khong tim thay client secret: {CLIENT_SECRET_FILE}")

    creds = None
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Token het han, dang refresh...")
            creds.refresh(Request())
        else:
            logger.info("Chua co token hop le, mo trinh duyet de dang nhap...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_FILE), DRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)

    return creds


# ===== upload backup len drive =====
def upload_to_drive(local_path: Path):
    """Tra ve (file_id, service) de tai su dung service cho buoc rotation."""
    credentials = get_credentials()
    service = build("drive", "v3", credentials=credentials)
    file_metadata = {
        "name": local_path.name,
        "parents": [DRIVE_FOLDER_ID],
    }
    media = MediaFileUpload(str(local_path), mimetype="application/x-sqlite3", resumable=True)
    uploaded = service.files().create(
        body=file_metadata, media_body=media, fields="id"
    ).execute()
    return uploaded.get("id"), service


# ===== xoa backup cu (local) =====
def rotate_local_backups() -> None:
    backups = sorted(LOCAL_BACKUP_DIR.glob("state_*.db"))
    if len(backups) > MAX_LOCAL_BACKUPS:
        for old_backup in backups[:-MAX_LOCAL_BACKUPS]:
            try:
                old_backup.unlink()
                logger.info("Da xoa backup cu (local): %s", old_backup)
            except OSError:
                logger.exception("Khong xoa duoc backup cu (local): %s", old_backup)


# xoa backup cu (google drive)
def rotate_drive_backups(service) -> None:
    """Liet ke cac file backup trong folder Drive, sap xep theo thoi gian
    tao, xoa cac ban cu nhat neu vuot qua MAX_DRIVE_BACKUPS, chi giu lai
    cac ban gan nhat."""
    query = (
        f"'{DRIVE_FOLDER_ID}' in parents "
        f"and name contains 'state_' "
        f"and trashed = false"
    )
    try:
        results = service.files().list(
            q=query,
            fields="files(id, name, createdTime)",
            orderBy="createdTime",  # tang dan: cu nhat truoc, moi nhat sau
            pageSize=1000,
        ).execute()
    except Exception:
        logger.exception("Khong lay duoc danh sach file tren Drive de rotate")
        return

    files = results.get("files", [])
    if len(files) <= MAX_DRIVE_BACKUPS:
        return

    files_to_delete = files[:-MAX_DRIVE_BACKUPS]  # giu lai N ban moi nhat
    for f in files_to_delete:
        try:
            service.files().delete(fileId=f["id"]).execute()
            logger.info("Da xoa backup cu (Drive): %s (%s)", f["name"], f["id"])
        except Exception:
            logger.exception("Khong xoa duoc backup tren Drive: %s", f["name"])


#main
def run_backup() -> None:
    logger.info("Bat dau backup database")
    try:
        backup_path = backup_to_local()
    except Exception:
        logger.exception("Backup local that bai")
        return
    logger.info("Backup local thanh cong: %s", backup_path)

    if not verify(backup_path):
        logger.error("Backup loi intergrity check, KHONG upload len Drive: %s", backup_path)
        backup_path.unlink()
        return

    try:
        file_id, service = upload_to_drive(backup_path)
        logger.info("Upload Drive thanh cong, file_id = %s", file_id)
    except Exception:
        logger.exception("Upload Drive that bai, backup local van duoc giu lai: %s", backup_path)
        return

    rotate_local_backups()
    rotate_drive_backups(service)
    logger.info("Backup hoan tat")


if __name__ == "__main__":
    run_backup()