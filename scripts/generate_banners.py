import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cairosvg
import io

# ─── Official logo SVG path data (exact brand paths at 260x260) ──────────────

# Python logo – official two-snake shield
PYTHON_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 110 110">
<path d="M54.918 2.000C42.936 2.000 34.000 6.627 34.000 13.000
L34.000 22.000 L56.000 22.000 L56.000 25.000 L25.000 25.000
C18.627 25.000 14.000 33.936 14.000 45.918
C14.000 57.900 18.627 67.836 25.000 67.836
L32.000 67.836 L32.000 57.000
C32.000 49.627 37.627 44.000 45.000 44.000
L67.000 44.000 C73.627 44.000 78.000 39.373 78.000 32.836
L78.000 13.000 C78.000 6.627 69.064 2.000 57.082 2.000
C56.421 2.000 55.564 2.000 54.918 2.000 Z
M43.000 10.000 C44.657 10.000 46.000 11.343 46.000 13.000
C46.000 14.657 44.657 16.000 43.000 16.000
C41.343 16.000 40.000 14.657 40.000 13.000
C40.000 11.343 41.343 10.000 43.000 10.000 Z"
fill="white"/>
<path d="M82.000 25.000 L82.000 35.836
C82.000 43.209 76.373 49.000 69.000 49.000
L47.000 49.000 C40.373 49.000 34.000 54.627 34.000 61.164
L34.000 81.000 C34.000 87.373 43.209 92.000 55.082 92.000
C67.064 92.000 76.000 87.373 76.000 81.000
L76.000 72.000 L54.000 72.000 L54.000 69.000 L85.000 69.000
C91.373 69.000 96.000 60.064 96.000 48.082
C96.000 36.000 91.373 27.000 85.000 27.000
L82.000 25.000 Z
M67.000 78.000 C68.657 78.000 70.000 79.343 70.000 81.000
C70.000 82.657 68.657 84.000 67.000 84.000
C65.343 84.000 64.000 82.657 64.000 81.000
C64.000 79.343 65.343 78.000 67.000 78.000 Z"
fill="white"/>
</svg>'''

# JavaScript logo – official JS badge with correct J and S letterforms
JS_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 260">
  <rect x="5" y="5" width="250" height="250" fill="white"/>
  <!-- J: thin vertical bar + bottom-left L hook (all rects, no bezier) -->
  <rect x="108" y="95"  width="24" height="115" fill="black"/>
  <rect x="68"  y="188" width="64" height="22"  fill="black"/>
  <rect x="68"  y="166" width="24" height="44"  fill="black"/>
  <!-- S: classic block S — two C shapes mirrored (all rects) -->
  <rect x="158" y="95"  width="68" height="22" fill="black"/>
  <rect x="158" y="95"  width="24" height="60" fill="black"/>
  <rect x="158" y="143" width="68" height="22" fill="black"/>
  <rect x="202" y="143" width="24" height="60" fill="black"/>
  <rect x="158" y="191" width="68" height="22" fill="black"/>
</svg>'''

# GitHub mark – official Octocat silhouette (verified correct ✅)
GITHUB_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 260">
  <path fill="white" d="M130,20 C70,20 20,70 20,132 C20,182 52,224 96,239
    C101,240 103,237 103,234 L103,214 C73,221 67,200 67,200
    C62,187 55,183 55,183 C45,176 56,177 56,177 C67,178 73,189 73,189
    C83,206 99,201 106,198 C107,191 110,186 113,183
    C88,180 62,170 62,128 C62,116 66,106 73,98 C72,95 68,83 74,67
    C74,67 83,64 103,78 C111,76 120,75 130,75 C140,75 149,76 157,78
    C177,64 186,67 186,67 C192,83 188,95 187,98 C194,106 198,116 198,128
    C198,170 172,180 147,183 C151,187 155,195 155,207 L155,234
    C155,237 157,240 162,239 C206,224 240,182 240,132
    C240,70 190,20 130,20 Z"/>
</svg>'''

def svg_to_pts(svg_str, target_n, size=260):
    """Rasterize an SVG string to a binary mask and extract filled pixel coordinates."""
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=size, output_height=size)
    img = Image.open(io.BytesIO(png_bytes)).convert('L')
    arr = np.array(img)
    pts = np.argwhere(arr > 128)  # (y, x)
    idx = np.random.choice(len(pts), target_n, replace=(len(pts) < target_n))
    return pts[idx]


def build_banner(mode='dark'):
    avatar_path = '/Users/novice/Desktop/Github/RBCs-lang/assets/avatar.png'
    img = Image.open(avatar_path).convert('RGB')

    w, h = img.size
    img = img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
    img = img.resize((60, 68), Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))

    gray = np.array(img.convert('L'), dtype=float)
    h_g, w_g = gray.shape

    mask = (gray < 240)
    dithered = np.zeros((h_g, w_g), dtype=int)
    err = gray.copy()

    for y in range(h_g):
        reverse = (y % 2 == 1)
        x_range = range(w_g - 1, -1, -1) if reverse else range(w_g)
        for x in x_range:
            old_val = err[y, x]
            new_val = 255 if old_val > 128 else 0
            dithered[y, x] = new_val
            error = old_val - new_val

            if not reverse:
                if x + 1 < w_g: err[y, x + 1] += error * 7 / 16.0
                if y + 1 < h_g:
                    if x > 0: err[y + 1, x - 1] += error * 3 / 16.0
                    err[y + 1, x] += error * 5 / 16.0
                    if x + 1 < w_g: err[y + 1, x + 1] += error * 1 / 16.0
            else:
                if x - 1 >= 0: err[y, x - 1] += error * 7 / 16.0
                if y + 1 < h_g:
                    if x + 1 < w_g: err[y + 1, x + 1] += error * 3 / 16.0
                    err[y + 1, x] += error * 5 / 16.0
                    if x - 1 >= 0: err[y + 1, x - 1] += error * 1 / 16.0

    ox, oy = 60, 160
    dw, dh = 5, 5

    portrait_pts = []
    for y in range(h_g):
        for x in range(w_g):
            if dithered[y, x] == 255 and mask[y, x]:
                portrait_pts.append((ox + x * dw, oy + y * dh))

    P0 = np.array(portrait_pts)
    N_DOTS = len(P0)
    print(f"Portrait particles: {N_DOTS}")

    np.random.seed(42)

    # Rasterize official SVG logos to point clouds, centered in VISUAL.MAP (ox=90, oy=170)
    LOGO_SIZE = 260
    logo_ox, logo_oy = 90, 170

    py_raw = svg_to_pts(PYTHON_SVG, N_DOTS, LOGO_SIZE)
    js_raw = svg_to_pts(JS_SVG,     N_DOTS, LOGO_SIZE)
    gh_raw = svg_to_pts(GITHUB_SVG, N_DOTS, LOGO_SIZE)

    P1 = np.column_stack([logo_ox + py_raw[:, 1], logo_oy + py_raw[:, 0]])
    P2 = np.column_stack([logo_ox + js_raw[:, 1], logo_oy + js_raw[:, 0]])
    P3 = np.column_stack([logo_ox + gh_raw[:, 1], logo_oy + gh_raw[:, 0]])

    # Optimal Transport: Hungarian algorithm for minimum-cost 1-to-1 particle matching
    def match(A, B):
        cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        _, col = linear_sum_assignment(cost)
        return B[col]

    P1m = match(P0, P1)
    P2m = match(P1m, P2)
    P3m = match(P2m, P3)

    particle_color = "#A78BFA" if mode == 'dark' else "#7C3AED"
    chrome_color   = "#22D3EE" if mode == 'dark' else "#0891B2"
    accent_color   = "#10B981"
    bg_color       = "#0A101F" if mode == 'dark' else "#F8FAFC"
    card_bg        = "#0F172A" if mode == 'dark' else "#FFFFFF"
    text_muted     = "#94A3B8" if mode == 'dark' else "#64748B"
    text_main      = "#F8FAFC" if mode == 'dark' else "#0F172A"

    # 16 s loop: 2.5 s hold + 1.5 s morph × 4 phases
    kt = "0; 0.1563; 0.25; 0.4063; 0.50; 0.6563; 0.75; 0.9063; 1"

    rects = []
    for i in range(N_DOTS):
        x0, y0 = int(P0[i,0]),  int(P0[i,1])
        x1, y1 = int(P1m[i,0]), int(P1m[i,1])
        x2, y2 = int(P2m[i,0]), int(P2m[i,1])
        x3, y3 = int(P3m[i,0]), int(P3m[i,1])
        vx = f"{x0};{x0};{x1};{x1};{x2};{x2};{x3};{x3};{x0}"
        vy = f"{y0};{y0};{y1};{y1};{y2};{y2};{y3};{y3};{y0}"
        ax = f'<animate attributeName="x" values="{vx}" keyTimes="{kt}" dur="16s" repeatCount="indefinite"/>'
        ay = f'<animate attributeName="y" values="{vy}" keyTimes="{kt}" dur="16s" repeatCount="indefinite"/>'
        rects.append(f'<rect x="{x0}" y="{y0}" width="4" height="4" rx="0.8" fill="{particle_color}">{ax}{ay}</rect>')

    morph_html = "\n    ".join(rects)

    info_rows = [
        ("Subject",       "Subh Sharma"),
        ("Role",          "Frontend Engineer"),
        ("Origin",        "Sonipat, Haryana, India"),
        ("Education",     "B.Tech (CS, DS &amp; Business)"),
        ("Status",        "Building + Learning + Shipping"),
        ("ToolChain",     "VS Code · Git · Android Studio · Figma"),
        ("Core.Lang",     "Python · JavaScript · HTML5 · CSS3"),
        ("Core.Frontend", "React · Next.js · Vite · Tailwind"),
        ("Core.Backend",  "Node.js · Express · REST APIs"),
        ("Core.Database", "MongoDB · PostgreSQL · SQLite"),
        ("Core.Infra",    "Git · GitHub · Vercel · Electron"),
        ("Grid.Mail",     "Available on Request"),
        ("Grid.Portfolio","Coming Soon"),
        ("Grid.LinkedIn", "linkedin.com/in/subh-sharma-86bb88368"),
        ("Grid.GitHub",   "github.com/RBCs"),
    ]

    rows_html = ""
    for idx, (label, val) in enumerate(info_rows):
        y = 145 + idx * 25
        rows_html += f'''
        <g transform="translate(0,{y})">
          <text x="440" fill="{text_muted}" font-family="Menlo,Monaco,'Courier New',monospace" font-size="11.5" font-weight="500">{label}</text>
          <path d="M560-4 h420" stroke="{text_muted}" stroke-dasharray="2 4" stroke-width="1" opacity="0.3"/>
          <text x="1120" text-anchor="end" fill="{text_main}" font-family="Menlo,Monaco,'Courier New',monospace" font-size="11.5" font-weight="600" textLength="340" lengthAdjust="spacingAndGlyphs">{val}</text>
        </g>'''

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">
  <style>
    .bg{{fill:{bg_color}}} .card{{fill:{card_bg};stroke:rgba(255,255,255,0.1);stroke-width:1}}
    .tbar{{fill:#1E293B}} .tr{{fill:#EF4444}} .ty{{fill:#F59E0B}} .tg{{fill:#10B981}}
    .ttl{{fill:{text_muted};font-family:Menlo,Monaco,monospace;font-size:13px;font-weight:600}}
  </style>
  <rect width="1180" height="610" rx="16" class="bg"/>
  <rect x="20" y="20" width="1140" height="570" rx="12" class="card"/>
  <rect x="20" y="20" width="1140" height="42" rx="12" class="tbar"/>
  <rect x="20" y="50" width="1140" height="12" class="tbar"/>
  <circle cx="48" cy="41" r="6" class="tr"/>
  <circle cx="68" cy="41" r="6" class="ty"/>
  <circle cx="88" cy="41" r="6" class="tg"/>
  <text x="590" y="45" text-anchor="middle" class="ttl">profile.sh --live</text>
  <g transform="translate(1010,33)">
    <rect width="64" height="20" rx="10" fill="rgba(239,68,68,0.15)" stroke="#EF4444" stroke-width="1"/>
    <circle cx="12" cy="10" r="4" class="tr"/>
    <text x="24" y="14" fill="#EF4444" font-family="Menlo,Monaco,monospace" font-size="11" font-weight="700">LIVE</text>
  </g>
  <g transform="translate(910,33)">
    <rect width="85" height="20" rx="10" fill="rgba(34,211,238,0.15)" stroke="{chrome_color}" stroke-width="1"/>
    <text x="42.5" y="14" text-anchor="middle" fill="{chrome_color}" font-family="Menlo,Monaco,monospace" font-size="11" font-weight="700">@RBCs</text>
  </g>
  <rect x="40" y="85" width="360" height="480" rx="8" fill="rgba(15,23,42,0.6)" stroke="{chrome_color}" stroke-width="1" opacity="0.8"/>
  <text x="55" y="112" fill="{chrome_color}" font-family="Menlo,Monaco,monospace" font-size="12" font-weight="700" letter-spacing="1">VISUAL.MAP</text>
  <g shape-rendering="crispEdges">
    {morph_html}
  </g>
  <text x="440" y="112" fill="{accent_color}" font-family="Menlo,Monaco,monospace" font-size="13" font-weight="700" letter-spacing="1">SYSTEM.INFO</text>
  <line x1="440" y1="122" x2="1120" y2="122" stroke="{accent_color}" stroke-width="1" opacity="0.3"/>
  {rows_html}
</svg>'''

    out = f'/Users/novice/Desktop/Github/RBCs-lang/assets/{mode}.svg'
    with open(out, 'w') as f:
        f.write(svg)
    print(f"Written → {out}")


build_banner('dark')
build_banner('light')
