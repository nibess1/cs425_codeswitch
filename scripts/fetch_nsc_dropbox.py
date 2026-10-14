#!/usr/bin/env python3
"""Mirror a public Dropbox shared folder (NSC Part 4) without the Dropbox API.

Usage:
  fetch_nsc_dropbox.py list  OUT_DIR      # crawl, write OUT_DIR/manifest.json
  fetch_nsc_dropbox.py fetch OUT_DIR [-j N] [--only SUBSTR]   # download (resumable)
"""
import argparse, http.cookiejar, json, os, subprocess, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SHARE = "https://www.dropbox.com/scl/fo/xo3a35ismqhts1yqutrc9/AKXiM2OqQ9cuiOf0WkNFnrI?rlkey=ofdc5pgqm8cslf61wmkeb6xcx&dl=0"
LINK_KEY, SECURE_HASH, RLKEY = "xo3a35ismqhts1yqutrc9", "AKXiM2OqQ9cuiOf0WkNFnrI", "ofdc5pgqm8cslf61wmkeb6xcx"
UA = "Mozilla/5.0"

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [("User-Agent", UA)]


def token():
    for c in cj:
        if c.name == "t":
            return c.value
    opener.open(SHARE).read()
    return token()


def list_dir(sub_path, secure_hash=SECURE_HASH):
    entries, voucher = [], None
    while True:
        t = token()
        form = {"is_xhr": "true", "t": t, "link_key": LINK_KEY, "link_type": "s",
                "secure_hash": secure_hash, "sub_path": sub_path, "rlkey": RLKEY}
        if voucher:
            form["voucher"] = voucher
        req = urllib.request.Request("https://www.dropbox.com/list_shared_link_folder_entries",
                                     data=urllib.parse.urlencode(form).encode(),
                                     headers={"X-CSRF-Token": t})
        for attempt in range(5):
            try:
                d = json.load(opener.open(req, timeout=60))
                break
            except Exception as e:
                print(f"retry {sub_path!r}: {e}", file=sys.stderr)
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"failed listing {sub_path!r}")
        entries += d["entries"]
        if not d.get("has_more_entries"):
            if len(entries) != d.get("total_num_entries", len(entries)):
                print(f"WARN {sub_path!r}: got {len(entries)} of {d['total_num_entries']}", file=sys.stderr)
            return entries
        voucher = d["next_request_voucher"]


def crawl(sub_path="", secure_hash=SECURE_HASH):
    files = []
    for e in list_dir(sub_path, secure_hash):
        p = f"{sub_path}/{e['filename']}"
        if e["is_dir"]:
            # each subfolder href is .../scl/fo/<link_key>/<its own secure hash>/<path>?...
            files += crawl(p, urllib.parse.urlparse(e["href"]).path.split("/")[4])
        else:
            files.append({"path": p.lstrip("/"), "bytes": e.get("bytes"), "href": e["href"]})
    print(f"listed {sub_path or '/'}: {len(files)} files so far", file=sys.stderr)
    return files


def fetch_one(f, out):
    dest = os.path.join(out, "raw", f["path"])
    if f["bytes"] and os.path.exists(dest) and os.path.getsize(dest) == f["bytes"]:
        return f["path"], "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f["href"].replace("dl=0", "dl=1")
    r = subprocess.run(["curl", "-sSL", "-A", UA, "--retry", "5", "--retry-delay", "5",
                        "-C", "-", "-o", dest, url], capture_output=True, text=True)
    if r.returncode not in (0, 33):  # 33 = range not supported -> retry fresh
        return f["path"], f"curl {r.returncode}: {r.stderr.strip()}"
    if f["bytes"] and os.path.getsize(dest) != f["bytes"]:
        os.remove(dest)
        return f["path"], "size mismatch (removed)"
    return f["path"], "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch"])
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(a.out, "manifest.json")
    if a.cmd == "list":
        files = crawl()
        json.dump(files, open(man, "w"), indent=1)
        tot = sum(f["bytes"] or 0 for f in files)
        print(f"{len(files)} files, {tot / 1e9:.1f} GB -> {man}")
        return
    files = [f for f in json.load(open(man)) if a.only in f["path"]]
    done = bad = 0
    with ThreadPoolExecutor(a.j) as ex:
        for fut in as_completed([ex.submit(fetch_one, f, a.out) for f in files]):
            p, status = fut.result()
            done += 1
            if status not in ("ok", "skip"):
                bad += 1
                print(f"FAIL {p}: {status}", flush=True)
            if done % 50 == 0 or done == len(files):
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(files)} done, {bad} failed", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
