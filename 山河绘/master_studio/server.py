# -*- coding: utf-8 -*-
"""名家笔触工坊：本地网页服务（标准库 + Pillow + NumPy）。"""

import argparse
import copy
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from master_studio import profiles, scene, render, corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
JOBS = {}
JOBS_LOCK = threading.Lock()
CUSTOM_FILE = ROOT / "data" / "custom_profiles.json"
CUSTOM = {}
if CUSTOM_FILE.exists():
    try:
        CUSTOM = json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
    except Exception:
        CUSTOM = {}

VARIANTS = ["平远", "秋霁", "云起", "晚照", "寒林", "归舟"]
MOODS = [
    {"id": "clear", "name": "清远", "desc": "留白多一分，墨色退半步"},
    {"id": "vast", "name": "苍茫", "desc": "云气更重，山脚藏进雾里"},
    {"id": "heroic", "name": "雄浑", "desc": "墨更重，山更逼人"},
    {"id": "sparse", "name": "萧散", "desc": "干笔多一些，像不经意的秋山"},
]


def all_artists():
    return profiles.ARTISTS + list(CUSTOM.values())


def artist_by_id(ident):
    for a in all_artists():
        if a["id"] == ident:
            return a
    raise ValueError("未找到该名家模型。")


def public_artist(a):
    return {k: a.get(k) for k in ("id", "name", "style_name", "dynasty", "years", "work",
                                  "school", "tags", "intro", "theory", "sources", "default_scene",
                                  "fingerprint", "custom")}


def public_meta():
    return {"artists": [public_artist(a) for a in all_artists()],
            "scenes": profiles.SCENES,
            "variants": VARIANTS,
            "moods": MOODS,
            "version": "规则蒸馏 v1"}


def find_export_python():
    candidates = [os.environ.get("MASTER_EXPORT_PYTHON"), sys.executable,
                  r"C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe",
                  r"D:\Anaconda3\python.exe", r"C:\ProgramData\Anaconda3\python.exe",
                  shutil.which("python")]
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            p = subprocess.run([c, "-c", "import sys, cv2, numpy; assert sys.version_info >= (3, 10)"],
                               capture_output=True, timeout=12)
            if p.returncode == 0:
                return c
        except Exception:
            continue
    return None


def write_job(job_id, job):
    try:
        data = {k: v for k, v in job.items() if k != "trace"}
        (OUTPUT / job_id / "job.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def load_jobs():
    """服务重启后仍能打开已经生成过的预览。"""
    if not OUTPUT.exists():
        return
    for d in OUTPUT.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "job.json"
        if meta_path.exists():
            try:
                job = json.loads(meta_path.read_text(encoding="utf-8"))
                if job.get("status") == "running":
                    job["status"] = "done"
                    job["stage"] = "完成（服务已重启）"
                JOBS[job["id"]] = job
                continue
            except Exception:
                pass
        previews = []
        for i in range(1, 7):
            f = d / ("preview-%d.png" % i)
            if f.exists():
                previews.append(dict(index=i - 1, label=VARIANTS[(i - 1) % len(VARIANTS)],
                                     url="/preview/%s/preview-%d.png" % (d.name, i),
                                     width=0, height=0, strokes=0))
        if previews:
            JOBS[d.name] = dict(id=d.name, status="done", stage="已恢复", done=len(previews),
                                total=len(previews), previews=previews, artist="huang_gongwang",
                                scene="fuchun", mood="clear")


def apply_mood(profile, mood):
    prof = copy.deepcopy(profile)
    if mood == "clear":
        prof["composition"]["empty"] = min(.95, prof["composition"]["empty"] + .10)
        prof["wash"]["mist"] = max(.05, prof["wash"]["mist"] - .10)
        prof["wash"]["saturation"] = max(.02, prof["wash"]["saturation"] - .10)
        prof["texture"]["density"] = max(.05, prof["texture"]["density"] - .12)
    elif mood == "vast":
        prof["wash"]["mist"] = min(1., prof["wash"]["mist"] + .25)
        prof["wash"]["opacity"] = min(.35, prof["wash"]["opacity"] + .04)
        prof["texture"]["density"] = max(.05, prof["texture"]["density"] - .06)
    elif mood == "heroic":
        prof["line"]["width"] = prof["line"]["width"] * 1.15
        prof["line"]["alpha"] = min(255, prof["line"]["alpha"] + 12)
        prof["dots"]["density"] = min(.98, prof["dots"]["density"] + .10)
        prof["composition"]["empty"] = max(.06, prof["composition"]["empty"] - .10)
        prof["texture"]["density"] = min(.99, prof["texture"]["density"] + .08)
    elif mood == "sparse":
        prof["line"]["dry"] = min(.85, prof["line"]["dry"] + .15)
        prof["texture"]["density"] = max(.04, prof["texture"]["density"] - .22)
        prof["dots"]["density"] = max(.03, prof["dots"]["density"] - .18)
        prof["composition"]["empty"] = min(.95, prof["composition"]["empty"] + .08)
    return prof


def apply_mood_layout(lay, mood):
    lay = copy.deepcopy(lay)
    water = float(lay.get("water", 0.70))
    mids = [m for m in lay["mountains"] if m["depth"] != "far"]
    if mood == "clear":
        lay["water"] = min(0.86, water + 0.025)
        for m in lay["mountains"]:
            m["height"] *= 0.94
    elif mood == "vast":
        lay["water"] = min(0.87, water + 0.045)
        for m in lay["mountains"]:
            m["base"] = min(1.10, m["base"] + 0.012)
        lay["mountains"].append(scene.M(0.06, lay["water"] + 0.01, 0.10, 0.18, "far", 0.20, 0.12))
        lay["mountains"].append(scene.M(0.94, lay["water"] + 0.01, 0.085, 0.16, "far", 0.18, 0.10))
    elif mood == "heroic":
        lay["water"] = max(0.45, water - 0.02)
        for m in lay["mountains"]:
            m["height"] *= 1.12
            m["base"] = min(1.10, m["base"] + 0.01)
    elif mood == "sparse":
        if len(mids) > 2:
            lay["mountains"].remove(min(mids, key=lambda m: m["height"]))
        if len(lay["trees"]) > 1:
            lay["trees"].pop()
        lay["water"] = min(0.87, water + 0.03)
        for m in lay["mountains"]:
            m["height"] *= 0.90
            m["width"] *= 0.94
    lay["mood"] = mood
    return lay


def run_generate(job_id, artist_id, scene_id, count, width, detail, mood="clear", abstraction=0.0, line_only=False):
    job = JOBS[job_id]
    try:
        prof = apply_mood(artist_by_id(artist_id), mood)
        prof = corpus.apply_model(prof)
        base = scene.layout_for(scene_id, 0)
        for i in range(count):
            job["stage"] = "正在生成第 %d / %d 幅" % (i + 1, count)
            lay = scene.layout_for(scene_id, i)
            lay = apply_mood_layout(lay, mood)
            label = lay.get("variant_name") or VARIANTS[i % len(VARIANTS)]
            # 与纸张 1.435:1 一致，预览与机械臂落纸不产生拉伸。
            lay["ratio"] = (1435, 1000)
            img, plan = render.render(prof, lay, seed=1000 + i * 37 + sum(map(ord, artist_id)) % 97,
                                      width=width, detail=detail, abstraction=abstraction,
                                      line_only=line_only)
            name = "preview-%d.png" % (i + 1)
            img.save(OUTPUT / job_id / name, quality=94)
            (OUTPUT / job_id / ("plan-%d.json" % (i + 1))).write_text(
                json.dumps(plan, ensure_ascii=False), encoding="utf-8")
            job["previews"].append(dict(index=i, label=label,
                                        url="/preview/%s/preview-%d.png" % (job_id, i + 1),
                                        width=img.width, height=img.height,
                                        strokes=len(plan)))
            job["done"] = i + 1
            write_job(job_id, job)
        job["status"] = "done"
        job["stage"] = "完成"
        write_job(job_id, job)
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["trace"] = traceback.format_exc()[-3000:]


class Handler(BaseHTTPRequestHandler):
    server_version = "ShanheMaster/1.0"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, content_type=None):
        path = Path(path)
        if not path.exists():
            self.send_json({"error": "文件不存在。"}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/new"):
                self.send_file(STATIC / "new.html", "text/html; charset=utf-8")
            elif path == "/classic":
                self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
            elif path == "/app.js":
                self.send_file(STATIC / "app.js", "application/javascript; charset=utf-8")
            elif path == "/style.css":
                self.send_file(STATIC / "style.css", "text/css; charset=utf-8")
            elif path == "/api/meta":
                self.send_json(public_meta())
            elif path.startswith("/corpus-image/") or path.startswith("/corpus-patch/"):
                parts = path.strip("/").split("/")
                if len(parts) != 3 or ".." in parts[1] or ".." in parts[2]:
                    raise ValueError("图库地址无效。")
                kind, aid, name = parts
                base = corpus.artist_dir(aid) / ("patches" if kind == "corpus-patch" else "")
                self.send_file(base / name, "image/png" if name.endswith(".png") else "image/jpeg")
            elif path == "/new":
                self.send_file(STATIC / "new.html", "text/html; charset=utf-8")
            elif path == "/new.css":
                self.send_file(STATIC / "new.css", "text/css; charset=utf-8")
            elif path == "/new.js":
                self.send_file(STATIC / "new.js", "application/javascript; charset=utf-8")
            elif path.startswith("/tree-preview/"):
                name = path.strip("/").split("/")[-1]
                if ".." in name: raise ValueError("地址无效。")
                self.send_file(OUTPUT / "trees" / name, "image/png")
            elif path.startswith("/flower-preview/"):
                name = path.strip("/").split("/")[-1]
                if ".." in name: raise ValueError("地址无效。")
                self.send_file(OUTPUT / "flowers" / name, "image/png")
            elif path == "/stroke-field-sample.png":
                self.send_file(ROOT / "data" / "stroke_field_vector.png", "image/png")
            elif path == "/stroke-field-compare.jpg":
                self.send_file(ROOT / "data" / "stroke_field_compare.jpg", "image/jpeg")
            elif path == "/api/tree/gallery":
                import json as _json
                tm = ROOT / "data" / "tree_model.json"
                if not tm.exists():
                    raise ValueError("树模型尚未建立。")
                m = _json.loads(tm.read_text(encoding="utf-8"))
                cards = []
                names = ["松林", "杂树", "枯枝"]
                for c in range(3):
                    files = [t["file"] for t in m.get("templates", []) if t["cluster"] == c][:4]
                    cards.append(dict(id=c, name=names[c], files=[dict(file=f, url="/corpus-tree/" + f) for f in files]))
                self.send_json(dict(cards=cards, patches=len(list((ROOT / "corpus" / "trees" / "patches").glob("*.png")))))
            elif path.startswith("/corpus-tree/"):
                name = path.strip("/").split("/")[-1]
                if ".." in name:
                    raise ValueError("树法地址无效。")
                self.send_file(ROOT / "corpus" / "trees" / "patches" / name, "image/png")
            elif path == "/api/corpus/status":
                self.send_json(corpus.status())
            elif path == "/api/latest":
                cands = []
                for d in OUTPUT.iterdir():
                    if d.is_dir() and any(d.glob("preview-*.png")):
                        cands.append((max(f.stat().st_mtime for f in d.glob("preview-*.png")), d.name))
                if not cands:
                    self.send_json({"error": "还没有生成记录。"}, 404)
                else:
                    jid = max(cands)[1]
                    self.send_json(JOBS.get(jid, {"id": jid, "error": "记录已失效。"}))
            elif path == "/api/job":
                jid = parse_qs(parsed.query).get("id", [""])[0]
                job = JOBS.get(jid)
                if not job:
                    self.send_json({"error": "任务不存在。"}, 404)
                else:
                    self.send_json({k: v for k, v in job.items() if k not in ("trace",)})
            elif path.startswith("/demo/"):
                name = path.strip("/").split("/")[-1]
                if not name.endswith(".png") or ".." in name:
                    raise ValueError("示例地址无效。")
                self.send_file(OUTPUT / "demo" / name, "image/png")
            elif path.startswith("/preview/"):
                parts = path.strip("/").split("/")
                if len(parts) != 3:
                    raise ValueError("预览地址无效。")
                _, jid, name = parts
                if not name.startswith("preview-"):
                    name = "preview-" + name
                self.send_file(OUTPUT / jid / name, "image/png")
            elif path.startswith("/download/"):
                parts = path.strip("/").split("/")
                if len(parts) != 3:
                    raise ValueError("下载地址无效。")
                _, jid, name = parts
                self.send_file(OUTPUT / jid / name, "application/json")
            else:
                self.send_json({"error": "页面不存在。"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32 * 1024 * 1024:
                raise ValueError("请求无效。")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if parsed.path == "/api/generate":
                artist_id = body.get("artist")
                artist_by_id(artist_id)
                scene_id = body.get("scene") or artist_by_id(artist_id)["default_scene"]
                scene.layout_for(scene_id, 0)
                count = int(body.get("count", 4))
                width = int(body.get("width", 1200))
                detail = float(body.get("detail", 1.0))
                mood = body.get("mood", "clear")
                if mood not in [m["id"] for m in MOODS]:
                    raise ValueError("气韵选项无效。")
                abstraction = float(body.get("abstraction", 0.0))
                line_only = bool(body.get("line_only", False))
                if not 0.0 <= abstraction <= 0.85:
                    raise ValueError("抽象度超出范围。")
                if not 1 <= count <= 6 or not 700 <= width <= 1500 or not 0.6 <= detail <= 1.4:
                    raise ValueError("生成参数超出范围。")
                jid = uuid.uuid4().hex[:10]
                (OUTPUT / jid).mkdir(parents=True, exist_ok=True)
                JOBS[jid] = dict(id=jid, status="running", stage="准备笔墨", done=0,
                                 total=count, previews=[], artist=artist_id, scene=scene_id, mood=mood,
                                 abstraction=abstraction, line_only=line_only)
                threading.Thread(target=run_generate, args=(jid, artist_id, scene_id, count, width, detail, mood, abstraction, line_only),
                                 daemon=True).start()
                self.send_json({"id": jid})
            elif parsed.path == "/api/chat":
                from master_studio import chat as chat_mod
                self.send_json(chat_mod.reply(body.get("message", ""), body.get("state") or {}))
            elif parsed.path == "/api/tree/generate":
                from master_studio import tree_render
                kind = body.get("kind", "pine")
                path_, meta = tree_render.render_tree(kind, seed=int(body.get("seed", 0)) or None)
                self.send_json(meta)
            elif parsed.path == "/api/flower/generate":
                from master_studio import flower_render
                kind = body.get("kind", "plum")
                path_, meta = flower_render.render_flower(kind, seed=int(body.get("seed", 0)) or None)
                self.send_json(meta)
            elif parsed.path == "/api/export":
                jid = body.get("job", "")
                index = int(body.get("index", 0))
                job = JOBS.get(jid)
                if not job or job.get("status") != "done":
                    raise ValueError("请先生成并选择一幅预览。")
                if not 0 <= index < len(job["previews"]):
                    raise ValueError("预览序号无效。")
                image = OUTPUT / jid / ("preview-%d.png" % (index + 1))
                plan = OUTPUT / jid / ("plan-%d.json" % (index + 1))
                py = find_export_python()
                if not py:
                    raise ValueError("未找到含 OpenCV 的 Python 3.10+ 环境，暂时无法导出机械臂包。")
                title = str(body.get("title") or "山河绘 · 名家笔触")[:50]
                result_path = OUTPUT / jid / ("painting-%d.result.json" % (index + 1))
                target = OUTPUT / jid / ("painting-%d.json" % (index + 1))
                cli = ROOT / "export_cli.py"
                proc = subprocess.run([py, "-u", str(cli), str(image), str(plan), str(target), title],
                                      cwd=str(PROJECT), capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", timeout=180,
                                      env=os.environ | {"PYTHONIOENCODING": "utf-8"})
                if not result_path.exists():
                    detail = (proc.stderr or proc.stdout or "导出失败")[-500:]
                    raise ValueError("导出失败：" + detail)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                self.send_json(dict(ok=True, download="/download/%s/painting-%d.json" % (jid, index + 1),
                                    black=result.get("black"), color=result.get("color"),
                                    waypoints=result.get("waypoints"), trajectory=result.get("trajectory")))
            elif parsed.path == "/api/corpus/download":
                raise ValueError("平台不代下载图片：请在可访问外网的机器运行 python -m master_studio.fetch_corpus --artist all --limit 4，或手动把图片放进 master_studio/corpus/<画家>/。")
            elif parsed.path == "/api/corpus/extract":
                m = corpus.manifest()
                counts = {}
                for aid, info in m["artists"].items():
                    if corpus.images_for(aid):
                        counts[aid] = corpus.extract_patches(aid, per_image=8, patch=240)
                self.send_json(dict(ok=True, counts=counts, status=corpus.status()))
            elif parsed.path == "/api/corpus/train":
                model = corpus.train_model()
                self.send_json(dict(ok=True, model=model, status=corpus.status()))
            elif parsed.path == "/api/install":
                jid = body.get("job", "")
                index = int(body.get("index", 0))
                job = JOBS.get(jid)
                if not job or job.get("status") != "done":
                    raise ValueError("请先生成并选择一幅预览。")
                src_json = OUTPUT / jid / ("painting-%d.json" % (index + 1))
                src_img = OUTPUT / jid / ("preview-%d.png" % (index + 1))
                if not src_json.exists() or not src_img.exists():
                    raise ValueError("请先导出绘画包，再保存到待画作品。")
                title = str(body.get("title") or "山河绘 · 名家笔触")[:50]
                safe = "".join(c for c in title if c not in '<>:"/\\|?*').strip(" .") or "名家笔触"
                target = PROJECT / "待画作品" / "新作品" / (safe + "_" + uuid.uuid4().hex[:6])
                target.mkdir(parents=True, exist_ok=False)
                artwork_id = "studio-" + uuid.uuid4().hex[:12]
                data = json.loads(src_json.read_text(encoding="utf-8"))
                ref = target / "参考原图.png"
                shutil.copy2(src_img, ref)
                data["workflow"]["reference_image"] = str(ref)
                data["ink_job"]["meta"]["image"] = str(ref)
                (target / "painting.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                (target / "制作参数.json").write_text(json.dumps(dict(
                    artworkId=artwork_id, title=title, settings={},
                    counts=dict(black=data["workflow"]["modules"]["outline"]["stroke_count"],
                                color=data["workflow"]["modules"]["color"]["stroke_count"])),
                    ensure_ascii=False), encoding="utf-8")
                self.send_json(dict(ok=True, path=str(target.relative_to(PROJECT)), artworkId=artwork_id,
                                    message="已保存到待画作品/新作品。重启 8765 工作台后出现在画库。"))
            elif parsed.path == "/api/train":
                from master_studio import learning
                profile = learning.train(body)
                CUSTOM[profile["id"]] = profile
                CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
                CUSTOM_FILE.write_text(json.dumps(CUSTOM, ensure_ascii=False, indent=2), encoding="utf-8")
                self.send_json({"artist": public_artist(profile)})
            else:
                self.send_json({"error": "接口不存在。"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


load_jobs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://127.0.0.1:%d" % args.port
    print("山河绘 · 名家笔触工坊：" + url, flush=True)
    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()