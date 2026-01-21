from dotenv import load_dotenv
import os
import requests
from typing import Iterable, Optional

messageIds = [
    "019b2a64-eebc-7815-84c5-e384fabc495c", #ictトラブルシューティング予選結果
    "019a48ff-b175-783c-93fc-b4ec6731dde5", #TechBookの販売記録
    "019bb6b8-5c57-7d1f-8b93-297c8e6c7827", #ポータブルモニターを購入
    "019bb5c0-5409-7b3b-ba9b-07449030364a", #大学の課題の締め切り
    "019bd043-d86c-79b8-9cf1-11351a482267", #pcの空き容量の画像
    "0197c4ef-8be2-792e-922b-c498b6948775", #traqの9点リーダーのアイコン
    "01963951-0f55-75b5-a53b-07f467ad9051", #sysad体験会
    "0195be37-dea6-7902-8d04-e46185873108", #githubへの招待
]

def _get_session() -> requests.Session:
    load_dotenv()
    r_session = os.getenv("r_session")
    if not r_session:
        raise RuntimeError("r_session が見つかりません（.env を確認してください）")

    session = requests.Session()
    session.cookies.set("r_session", r_session)
    return session


def get_messages(target_message_ids: Optional[Iterable[str]] = None):
    base_url = "https://q.trap.jp/api/v3"
    ids = list(target_message_ids) if target_message_ids is not None else messageIds

    result = []

    with _get_session() as session:
        for message_id in ids:
            url = f"{base_url}/messages/{message_id}"

            try:
                response = session.get(url)
                response.raise_for_status()
                result.append(response.json())
            except requests.exceptions.RequestException as e:
                print(f"Error fetching message {message_id}: {e}")

    return result


def download_file(file_id: str, base_url: str = "https://q.trap.jp/api/v3") -> bytes:
    with _get_session() as session:
        url = f"{base_url}/files/{file_id}/raw"
        response = session.get(url)
        response.raise_for_status()
        return response.content
                
