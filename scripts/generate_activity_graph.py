import urllib.request
import ssl
import re
import os
import json
import math
from pathlib import Path

def fetch_contributions(username="RBCs-lang"):
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    
    # Try GraphQL API if token exists
    if token:
        try:
            query = """
            query($login: String!) {
              user(login: $login) {
                contributionsCollection {
                  contributionCalendar {
                    weeks {
                      contributionDays {
                        date
                        contributionCount
                      }
                    }
                  }
                }
              }
            }
            """
            req_data = json.dumps({"query": query, "variables": {"login": username}}).encode("utf-8")
            req = urllib.request.Request(
                "https://api.github.com/graphql",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "GitHub-Activity-Graph-Generator",
                    "Content-Type": "application/json"
                }
            )
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                weeks = result["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]
                days = []
                for w in weeks:
                    for d in w["contributionDays"]:
                        days.append((d["date"], d["contributionCount"]))
                days.sort(key=lambda x: x[0])
                if days:
                    return days[-31:]
        except Exception as e:
            print(f"GraphQL fetch failed ({e}), falling back to web scraping...")

    # Fallback to scraping public contributions page
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"https://github.com/users/{username}/contributions"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        html = resp.read().decode("utf-8")

    td_matches = re.findall(r'<td[^>]+data-date=\"([0-9-]+)\"[^>]+id=\"([^\"]+)\"', html)
    tooltip_matches = dict(re.findall(r'<tool-tip[^>]+for=\"([^\"]+)\"[^>]*>(.*?)</tool-tip>', html, re.DOTALL))

    data = []
    for date, comp_id in td_matches:
        tip = tooltip_matches.get(comp_id, "")
        cnt_m = re.search(r'([0-9,]+)\s+contribution', tip)
        if cnt_m:
            cnt = int(cnt_m.group(1).replace(",", ""))
        elif "No contribution" in tip:
            cnt = 0
        else:
            cnt = 0
        data.append((date, cnt))

    data.sort(key=lambda x: x[0])
    return data[-31:]


def get_smooth_path(points):
    """Generate smooth cubic Bezier curve path string for a list of (x, y) points."""
    if not points:
        return ""
    if len(points) == 1:
        return f"M {points[0][0]},{points[0][1]}"

    path = [f"M {points[0][0]:.2f},{points[0][1]:.2f}"]
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else p2

        # Catmull-Rom to Cubic Bezier conversion
        cp1x = p1[0] + (p2[0] - p0[0]) / 6.0
        cp1y = p1[1] + (p2[1] - p0[1]) / 6.0
        cp2x = p2[0] - (p3[0] - p1[0]) / 6.0
        cp2y = p2[1] - (p3[1] - p1[1]) / 6.0

        path.append(f"C {cp1x:.2f},{cp1y:.2f} {cp2x:.2f},{cp2y:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(path)


def generate_activity_svg(days_data, username="RBCs-lang", theme="red"):
    width = 950
    height = 360

    # Theme definitions
    themes = {
        "red": {
            "bg": "#0A101F",
            "card_border": "rgba(255, 255, 255, 0.08)",
            "title": "#E2E8F0",
            "text_muted": "#64748B",
            "grid": "rgba(255, 255, 255, 0.07)",
            "line": "#EF4444",
            "point": "#EF4444",
            "area_top": "rgba(239, 68, 68, 0.35)",
            "area_bottom": "rgba(239, 68, 68, 0.00)"
        },
        "cyan": {
            "bg": "#0A101F",
            "card_border": "rgba(255, 255, 255, 0.08)",
            "title": "#E2E8F0",
            "text_muted": "#64748B",
            "grid": "rgba(255, 255, 255, 0.07)",
            "line": "#22D3EE",
            "point": "#10B981",
            "area_top": "rgba(34, 211, 238, 0.35)",
            "area_bottom": "rgba(34, 211, 238, 0.00)"
        }
    }
    th = themes.get(theme, themes["red"])

    # Graph plotting boundaries
    graph_left = 95
    graph_right = width - 40
    graph_top = 75
    graph_bottom = height - 55
    plot_width = graph_right - graph_left
    plot_height = graph_bottom - graph_top

    counts = [c for _, c in days_data]
    max_c = max(counts) if counts else 10
    if max_c <= 0:
        max_c = 10

    # Calculate nice Y-axis scale (e.g. 25, 50, 75, 100 or 10, 20, 30...)
    raw_step = max_c / 4.0
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    residual = raw_step / magnitude
    if residual <= 1.5:
        step = 1 * magnitude
    elif residual <= 3.0:
        step = 2.5 * magnitude
    elif residual <= 7.0:
        step = 5 * magnitude
    else:
        step = 10 * magnitude

    if step < 1:
        step = 1
    step = int(step) if step >= 1 else step

    num_ticks = math.ceil(max_c / step)
    if num_ticks < 3:
        num_ticks = 4
        step = math.ceil(max_c / num_ticks) or 1

    y_max = num_ticks * step

    # Calculate point coordinates
    n = len(days_data)
    points = []
    for i, (d_str, cnt) in enumerate(days_data):
        x = graph_left + (i / (n - 1) if n > 1 else 0.5) * plot_width
        y = graph_bottom - (cnt / y_max) * plot_height
        points.append((x, y))

    # Build line path and area path
    smooth_line = get_smooth_path(points)
    first_pt = points[0]
    last_pt = points[-1]
    area_d = f"{smooth_line} L {last_pt[0]:.2f},{graph_bottom:.2f} L {first_pt[0]:.2f},{graph_bottom:.2f} Z"

    # Y-axis ticks and grid lines
    grid_svg = []
    for t in range(num_ticks + 1):
        val = t * step
        y_pos = graph_bottom - (val / y_max) * plot_height
        grid_svg.append(f'''
        <line x1="{graph_left}" y1="{y_pos:.1f}" x2="{graph_right}" y2="{y_pos:.1f}" stroke="{th["grid"]}" stroke-dasharray="3 4" stroke-width="1"/>
        <text x="{graph_left - 15}" y="{y_pos + 4:.1f}" text-anchor="end" fill="{th["text_muted"]}" font-family="system-ui, -apple-system, sans-serif" font-size="12" font-weight="500">{int(val)}</text>
        ''')

    # X-axis ticks (day numbers)
    x_ticks_svg = []
    for i, (d_str, cnt) in enumerate(days_data):
        x = points[i][0]
        day_num = int(d_str.split("-")[2])
        x_ticks_svg.append(f'''
        <text x="{x:.1f}" y="{graph_bottom + 22}" text-anchor="middle" fill="{th["text_muted"]}" font-family="system-ui, -apple-system, sans-serif" font-size="11.5" font-weight="500">{day_num}</text>
        ''')

    # Circle points with tooltips
    points_svg = []
    for i, (d_str, cnt) in enumerate(days_data):
        x, y = points[i]
        points_svg.append(f'''
        <g class="graph-point">
          <circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{th["point"]}" stroke="{th["bg"]}" stroke-width="1.5">
            <title>{d_str}: {cnt} contributions</title>
          </circle>
        </g>
        ''')

    grad_id = f"areaGradient_{theme}"
    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
  <defs>
    <linearGradient id="{grad_id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{th["area_top"]}"/>
      <stop offset="100%" stop-color="{th["area_bottom"]}"/>
    </linearGradient>
  </defs>

  <style>
    .card-bg {{ fill: {th["bg"]}; stroke: {th["card_border"]}; stroke-width: 1; }}
    .title {{ fill: {th["title"]}; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; }}
    .axis-label {{ fill: {th["text_muted"]}; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; }}
    .graph-point circle {{ transition: r 0.2s ease, fill 0.2s ease; cursor: pointer; }}
    .graph-point circle:hover {{ r: 7; fill: #FFFFFF; filter: drop-shadow(0 0 6px {th["line"]}); }}
  </style>

  <!-- Background Card -->
  <rect width="{width}" height="{height}" rx="12" class="card-bg"/>

  <!-- Title -->
  <text x="{width / 2}" y="36" text-anchor="middle" class="title">{username}'s Contribution Graph</text>

  <!-- Y-Axis Label -->
  <text x="{- (graph_top + plot_height / 2)}" y="32" text-anchor="middle" transform="rotate(-90)" class="axis-label">Contributions</text>

  <!-- X-Axis Label -->
  <text x="{graph_left + plot_width / 2}" y="{height - 12}" text-anchor="middle" class="axis-label">Days</text>

  <!-- Grid & Ticks -->
  {"".join(grid_svg)}
  {"".join(x_ticks_svg)}

  <!-- Area Gradient Fill -->
  <path d="{area_d}" fill="url(#{grad_id})"/>

  <!-- Line Chart Path -->
  <path d="{smooth_line}" fill="none" stroke="{th["line"]}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Point Markers -->
  {"".join(points_svg)}
</svg>'''
    return svg_content


if __name__ == "__main__":
    username = "RBCs-lang"
    base_dir = Path(__file__).resolve().parent.parent
    assets_dir = base_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    print(f"Fetching contribution data for {username}...")
    data = fetch_contributions(username)
    print(f"Loaded {len(data)} days of contribution activity.")

    # Save Red (matching screenshot)
    red_svg = generate_activity_svg(data, username=username, theme="red")
    out_red = assets_dir / "activity-dark.svg"
    with open(out_red, "w", encoding="utf-8") as f:
        f.write(red_svg)
    print(f"Generated -> {out_red}")

    # Save Cyan (matching profile palette)
    cyan_svg = generate_activity_svg(data, username=username, theme="cyan")
    out_cyan = assets_dir / "activity-cyan.svg"
    with open(out_cyan, "w", encoding="utf-8") as f:
        f.write(cyan_svg)
    print(f"Generated -> {out_cyan}")
