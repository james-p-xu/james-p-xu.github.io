import os
import shutil
import datetime as dt
import pathlib
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
import mistune
from slugify import slugify

ROOT = pathlib.Path(__file__).parent.resolve()
SRC = ROOT / "src"
TEMPLATES = SRC / "templates"
ASSETS = SRC / "assets"
POSTS_DIR = SRC / "posts"
PAGES_DIR = SRC / "pages"
PUBLIC = ROOT / "public"

DATE_FMT = "%Y-%m-%d"


def read_site_config():
    config_path = ROOT / "site.yml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_env(site):
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.update(
        title=site.get("title", ""),
        description=site.get("description", ""),
        author=site.get("author", ""),
        base_url=site.get("base_url", ""),
        now=dt.datetime.utcnow(),
        rss=bool(site.get("rss", False)),
        social=site.get("social", {}),
    )
    return env


def coerce_date(value):
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    if isinstance(value, str):
        return dt.datetime.strptime(value, DATE_FMT)
    return dt.datetime.utcnow()


def parse_front_matter_md(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    meta = {}
    content_md = text
    if text.startswith("---\n"):
        _, rest = text.split("---\n", 1)
        fm, content_md = rest.split("\n---\n", 1)
        meta = yaml.safe_load(fm) or {}

    # MD renderer with header IDs
    class HeaderWithID(mistune.HTMLRenderer):
        def heading(self, text, level):
            # generate slugified ID
            id_attr = slugify(text)
            return f'<h{level} id="{id_attr}">{text}</h{level}>\n'

    md = mistune.create_markdown(
        renderer=HeaderWithID(),
        plugins=["strikethrough", "table", "task_lists", "url", "footnotes"]
    )
    html = md(content_md)
    return meta, html

def render_inline_markdown(text: str) -> str:
    renderer = mistune.HTMLRenderer()
    md = mistune.create_markdown(renderer=renderer)

    html = md(text)
    # Remove <p>...</p> if it's wrapping the whole string
    if html.startswith("<p>") and html.endswith("</p>\n"):
        html = html[3:-5]
    return html

def parse_post(path: pathlib.Path):
    meta, html = parse_front_matter_md(path)
    title = meta.get("title") or path.stem.replace("-", " ").title()
    date_val = meta.get("date")
    date = coerce_date(date_val) if date_val is not None else dt.datetime.utcnow()
    slug = meta.get("slug") or slugify(title)
    description = meta.get("description", "")
    references_md = meta.get("references", [])

    # Render Markdown inside references
    references_html = [render_inline_markdown(r) for r in references_md]

    # Replace [1], [2], etc. in html with links to footer
    for i, _ in enumerate(references_html, start=1):
        html = html.replace(f"[{i}]", f'<a href="#ref-{i}" class="reference">[{i}]</a>')

    return {
        "title": title,
        "date": date,
        "date_iso": date.date().isoformat(),
        "date_human": date.strftime("%b %d, %Y"),
        "slug": slug,
        "description": description,
        "html": html,
        "references": references_html,
        "source_path": str(path.relative_to(ROOT)),
    }


def parse_page(path: pathlib.Path):
    meta, html = parse_front_matter_md(path)
    title = meta.get("title") or path.stem.replace("-", " ").title()
    slug = meta.get("slug") or slugify(title)
    description = meta.get("description", "")
    return {
        "title": title,
        "slug": slug,
        "description": description,
        "html": html,
        "source_path": str(path.relative_to(ROOT)),
    }


def load_posts():
    posts = []
    if not POSTS_DIR.exists():
        return posts
    for path in sorted(POSTS_DIR.glob("*.md")):
        posts.append(parse_post(path))
    posts.sort(key=lambda p: p["date"], reverse=True)
    return posts


def load_pages():
    pages = []
    if not PAGES_DIR.exists():
        return pages
    for path in sorted(PAGES_DIR.glob("*.md")):
        pages.append(parse_page(path))
    return pages


def write_file(path: pathlib.Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


essential_files = [
    "index.html",
]


def build_site():
    site = read_site_config()
    env = create_env(site)

    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    if ASSETS.exists():
        shutil.copytree(ASSETS, PUBLIC / "assets")

    posts = load_posts()
    pages = load_pages()

    index_tpl = env.get_template("index.html")
    index_html = index_tpl.render(posts=posts, page_title=None, pages=pages)
    write_file(PUBLIC / "index.html", index_html)

    post_tpl = env.get_template("post.html")
    for post in posts:
        html = post_tpl.render(post=post, page_title=post["title"], pages=pages)
        write_file(PUBLIC / post["slug"] / "index.html", html)

    page_tpl = env.get_template("page.html")
    for page in pages:
        html = page_tpl.render(page=page, page_title=page["title"], pages=pages)
        write_file(PUBLIC / page["slug"] / "index.html", html)

    if site.get("rss", False):
        write_rss(site, posts)

    write_sitemap(site, posts, pages)

    print(f"Built {len(posts)} posts, {len(pages)} pages → {PUBLIC}")


def write_rss(site, posts):
    url = site.get("url", "").rstrip("/")
    items = []
    for p in posts:
        post_url = f"{url}/{p['slug']}/"
        items.append(f"""
  <item>
    <title>{escape_xml(p['title'])}</title>
    <link>{post_url}</link>
    <guid>{post_url}</guid>
    <pubDate>{p['date'].strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>
    <description><![CDATA[{p['description'] or p['html'][:280]}]]></description>
  </item>""")
    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>{escape_xml(site.get('title',''))}</title>
  <link>{url}/</link>
  <description>{escape_xml(site.get('description',''))}</description>
  {''.join(items)}
</channel>
</rss>
"""
    write_file(PUBLIC / "rss.xml", rss)


def write_sitemap(site, posts, pages):
    url = site.get("url", "").rstrip("/")
    urls = [f"{url}/"] + [f"{url}/{p['slug']}/" for p in posts] + [f"{url}/{pg['slug']}/" for pg in pages]
    xml = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"]
    for u in urls:
        xml.append("  <url><loc>" + u + "</loc></url>")
    xml.append("</urlset>")
    write_file(PUBLIC / "sitemap.xml", "\n".join(xml))


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


if __name__ == "__main__":
    build_site()
