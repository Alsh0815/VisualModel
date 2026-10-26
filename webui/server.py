import os
import sys
import json
import mimetypes
import argparse
import subprocess
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# 自作モジュールのインポート
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import tokenizer

DATASETS_DIR = PROJECT_ROOT / "datasets"
STATIC_DIR = CURRENT_DIR / "static"

# 非同期タスク管理（prepare_text.py用）
task_state = {
    "running": False,
    "subset": None,
    "logs": [],
    "returncode": None
}
task_lock = threading.Lock()


def get_image_info(img_path: Path):
    """画像ファイルのサイズ、解像度、アスペクト比を取得"""
    file_size = img_path.stat().st_size
    width, height = 0, 0
    if HAS_PIL:
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            pass
    aspect_ratio = f"{width}:{height}" if width and height else "-"
    ratio_float = round(width / height, 2) if width and height else 1.0
    return {
        "file_size": file_size,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "ratio_float": ratio_float
    }


def scan_dataset(subset_name: str):
    """指定されたサブセットフォルダから画像とプロンプト情報を抽出"""
    target_dir = DATASETS_DIR / subset_name
    if not target_dir.is_dir():
        return []

    # 1. captions.json の読み込み
    captions_json_data = {}
    json_path = target_dir / "captions.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                captions_json_data = json.load(f)
        except Exception as e:
            print(f"[Warning] captions.json parse error: {e}")

    # 2. 画像ファイルの走査
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    items = []

    for file_path in sorted(target_dir.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in image_extensions:
            base_name = file_path.stem
            txt_path = target_dir / f"{base_name}.txt"
            tensors_pt = target_dir / "tensors" / f"{base_name}.pt"
            
            # プロンプトの取得 (txt優先, 次にjson)
            prompt = ""
            has_txt = txt_path.exists()
            if has_txt:
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()
                except Exception:
                    pass
            elif base_name in captions_json_data:
                prompt = str(captions_json_data[base_name]).strip()
            elif file_path.name in captions_json_data:
                prompt = str(captions_json_data[file_path.name]).strip()

            img_info = get_image_info(file_path)

            items.append({
                "id": base_name,
                "filename": file_path.name,
                "base_name": base_name,
                "ext": ext,
                "has_txt": has_txt,
                "has_tensor": tensors_pt.exists(),
                "prompt": prompt,
                "char_count": len(prompt),
                "char_count_no_spaces": len("".join(prompt.split())),
                "mtime": file_path.stat().st_mtime,
                **img_info
            })

    return items


def list_subsets():
    """datasets ディレクトリ配下のサブフォルダ一覧と概要を取得"""
    if not DATASETS_DIR.exists():
        return []
    
    subsets = []
    for entry in sorted(DATASETS_DIR.iterdir()):
        if entry.is_dir():
            # 画像枚数とtxt数をカウント
            image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
            img_count = sum(1 for f in entry.iterdir() if f.is_file() and f.suffix.lower() in image_extensions)
            txt_count = sum(1 for f in entry.iterdir() if f.is_file() and f.suffix.lower() == ".txt")
            subsets.append({
                "name": entry.name,
                "image_count": img_count,
                "txt_count": txt_count
            })
    return subsets


class WebUIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        # CORS & キャッシュコントロール
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status=400):
        self.send_json({"error": message}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. API: サブセット一覧
        if path == "/api/subsets":
            subsets = list_subsets()
            self.send_json({"subsets": subsets})
            return

        # 2. API: 指定サブセットの全アイテム
        if path == "/api/dataset":
            subset = query.get("subset", [""])[0]
            if not subset:
                self.send_error_json("subset parameter is required")
                return
            items = scan_dataset(subset)
            self.send_json({
                "subset": subset,
                "total_items": len(items),
                "items": items
            })
            return

        # 3. API: 画像配信
        if path == "/api/image":
            subset = query.get("subset", [""])[0]
            filename = query.get("filename", [""])[0]
            if not subset or not filename:
                self.send_error_json("subset and filename parameters are required")
                return
            
            # ディレクトリトラバーサル防止
            safe_filename = Path(filename).name
            img_path = DATASETS_DIR / subset / safe_filename
            if not img_path.is_file():
                self.send_error_json("Image not found", status=404)
                return

            mime_type, _ = mimetypes.guess_type(str(img_path))
            if not mime_type:
                mime_type = "application/octet-stream"

            file_size = img_path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()

            with open(img_path, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)
            return

        # 4. API: 単語出現頻度統計
        if path == "/api/word_stats":
            subset = query.get("subset", [""])[0]
            min_length = int(query.get("min_length", ["1"])[0])
            filter_stopwords = query.get("filter_stopwords", ["true"])[0].lower() == "true"

            records = []
            if subset == "all" or not subset:
                # 全サブセット横断
                for s in list_subsets():
                    records.extend(scan_dataset(s["name"]))
            else:
                records = scan_dataset(subset)

            stats = tokenizer.calculate_word_frequencies(
                records,
                filter_stopwords=filter_stopwords,
                min_length=min_length
            )
            self.send_json(stats)
            return

        # 5. API: テンソル生成ステータス確認
        if path == "/api/prepare_status":
            with task_lock:
                self.send_json(task_state)
            return

        # 6. 静的ファイル (index.html, style.css, app.js 等)
        if path == "/" or path == "/index.html":
            file_to_serve = STATIC_DIR / "index.html"
        else:
            rel_path = path.lstrip("/")
            file_to_serve = STATIC_DIR / rel_path

        if file_to_serve.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_to_serve))
            if not mime_type:
                mime_type = "application/octet-stream"
            content = file_to_serve.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if "text" in mime_type or "javascript" in mime_type else mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error_json("Not Found", status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            self.send_error_json("Invalid JSON payload")
            return

        # 1. API: プロンプト保存
        if path == "/api/save_prompt":
            subset = body.get("subset", "")
            base_name = body.get("base_name", "")
            new_prompt = body.get("prompt", "")

            if not subset or not base_name:
                self.send_error_json("subset and base_name are required")
                return

            target_dir = DATASETS_DIR / subset
            if not target_dir.is_dir():
                self.send_error_json("Subset directory not found", status=404)
                return

            # 個別 .txt ファイルへ書き込み
            txt_path = target_dir / f"{base_name}.txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_prompt)
            except Exception as e:
                self.send_error_json(f"Failed to write file: {e}", status=500)
                return

            # captions.json が存在する場合は同期更新
            json_path = target_dir / "captions.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        captions_json = json.load(f)
                    captions_json[base_name] = new_prompt
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(captions_json, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[Warning] Failed to sync captions.json: {e}")

            self.send_json({
                "success": True,
                "message": f"Saved {base_name}.txt",
                "char_count": len(new_prompt),
                "char_count_no_spaces": len("".join(new_prompt.split()))
            })
            return

        # 2. API: 一括単語置換
        if path == "/api/batch_replace":
            subset = body.get("subset", "")
            find_word = body.get("find_word", "")
            replace_word = body.get("replace_word", "")
            preview_only = body.get("preview_only", False)

            if not find_word:
                self.send_error_json("find_word is required")
                return

            subsets_to_process = [subset] if subset and subset != "all" else [s["name"] for s in list_subsets()]
            modified_records = []

            for s_name in subsets_to_process:
                s_dir = DATASETS_DIR / s_name
                items = scan_dataset(s_name)
                for item in items:
                    prompt = item["prompt"]
                    if find_word in prompt:
                        new_p = prompt.replace(find_word, replace_word)
                        modified_records.append({
                            "subset": s_name,
                            "filename": item["filename"],
                            "base_name": item["base_name"],
                            "old_prompt": prompt,
                            "new_prompt": new_p,
                            "occurrences": prompt.count(find_word)
                        })
                        if not preview_only:
                            txt_path = s_dir / f"{item['base_name']}.txt"
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(new_p)

            self.send_json({
                "success": True,
                "preview_only": preview_only,
                "modified_count": len(modified_records),
                "changes": modified_records
            })
            return

        # 3. API: テンソル生成 (prepare_text.py 実行)
        if path == "/api/run_prepare_text":
            subset = body.get("subset", "")
            if not subset:
                self.send_error_json("subset is required")
                return

            with task_lock:
                if task_state["running"]:
                    self.send_error_json("Another prepare_text task is already running")
                    return
                task_state["running"] = True
                task_state["subset"] = subset
                task_state["logs"] = [f"Starting prepare_text.py for subset: {subset}..."]
                task_state["returncode"] = None

            def run_worker():
                try:
                    cmd = [sys.executable, str(PROJECT_ROOT / "prepare_text.py"), "--subset", subset]
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        cwd=str(PROJECT_ROOT)
                    )
                    for line in proc.stdout:
                        with task_lock:
                            task_state["logs"].append(line.rstrip())
                    proc.wait()
                    with task_lock:
                        task_state["running"] = False
                        task_state["returncode"] = proc.returncode
                        task_state["logs"].append(f"Completed with exit code: {proc.returncode}")
                except Exception as ex:
                    with task_lock:
                        task_state["running"] = False
                        task_state["returncode"] = -1
                        task_state["logs"].append(f"Error executing prepare_text.py: {ex}")

            threading.Thread(target=run_worker, daemon=True).start()
            self.send_json({"success": True, "message": "prepare_text task started"})
            return

        self.send_error_json("Not Found", status=404)


def run_server(host="127.0.0.1", port=7860, open_browser=True):
    # ポートが既に使用中の場合、利用可能な次のポートを自動検索
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, p), WebUIHandler)
            server.daemon_threads = True
            port = p
            break
        except OSError:
            continue

    if not server:
        print(f"Error: Unable to bind to {host}:{port} - {port+20}")
        sys.exit(1)

    url = f"http://{host}:{port}/"
    print("=" * 60)
    print("  VisualModel Dataset Studio WebUI")
    print(f"  URL: {url}")
    print(f"  Datasets root: {DATASETS_DIR}")
    print("  Press Ctrl+C to terminate the server.")
    print("=" * 60)

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisualModel Dataset Studio WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
