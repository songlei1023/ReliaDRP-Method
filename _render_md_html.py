# -*- coding: utf-8 -*-
"""Generic light-theme Markdown -> standalone HTML renderer.

Usage:  E:/anaconda/python.exe _render_md_html.py <src.md> [dst.html]
"""
import io, os, sys
import markdown

CSS = """
:root{
  --bg:#f7f8fa; --card:#ffffff; --ink:#1f2329; --ink-2:#4a5159; --ink-3:#7a828c;
  --line:#e3e6ea; --accent:#2b6cb0; --accent-soft:#eef4fb; --code-bg:#f4f6f8;
  --h1:#12263f; --h2:#1c3d5a; --h3:#28486b; --warn:#b3541e; --warn-bg:#fdf3ec;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;padding:56px 24px 96px;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",
              "Microsoft YaHei","Source Han Sans SC",sans-serif;
  font-size:15px;line-height:1.75;-webkit-font-smoothing:antialiased}
main{max-width:1120px;margin:0 auto;background:var(--card);border:1px solid var(--line);
  border-radius:14px;padding:48px 56px 64px;box-shadow:0 1px 3px rgba(16,24,40,.04),0 12px 32px rgba(16,24,40,.06)}
h1,h2,h3,h4{color:var(--h1);line-height:1.35;font-weight:650}
h1{font-size:28px;margin:0 0 18px;padding-bottom:14px;border-bottom:2px solid var(--accent)}
h2{font-size:21px;margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:17px;margin:30px 0 10px;color:var(--h2)}
h4{font-size:15px;margin:22px 0 8px;color:var(--h3)}
p{margin:10px 0}
ul,ol{margin:10px 0;padding-left:24px}
li{margin:5px 0}
strong{color:#101828;font-weight:650}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
hr{border:0;border-top:1px solid var(--line);margin:32px 0}
code{background:var(--code-bg);border:1px solid var(--line);border-radius:5px;
  padding:1px 5px;font-size:13px;font-family:"Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;color:#b02a37}
pre{background:#f8f9fb;border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow:auto}
pre code{background:none;border:0;color:var(--ink);padding:0}
blockquote{margin:14px 0;padding:10px 18px;background:var(--accent-soft);
  border-left:3px solid var(--accent);border-radius:0 8px 8px 0;color:var(--ink-2)}
blockquote p{margin:4px 0}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
th{background:#f1f4f8;color:var(--h2);font-weight:650;white-space:nowrap}
tbody tr:nth-child(even){background:#fbfcfd}
details{margin:14px 0;border:1px solid var(--line);border-radius:10px;background:#fcfdfe;padding:8px 14px}
summary{cursor:pointer;font-weight:600;color:var(--h2);padding:4px 0}
"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if not src:
        raise SystemExit("usage: _render_md_html.py <src.md> [dst.html]")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".html"
    with io.open(src, "r", encoding="utf-8") as f:
        text = f.read()
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        output_format="html5",
    )
    title = text.splitlines()[0].lstrip("# ").strip() if text else "report"
    html = ("<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n<main>\n%s\n</main>\n</body>\n</html>\n"
            % (title, CSS, body))
    with io.open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", dst)


if __name__ == "__main__":
    main()
