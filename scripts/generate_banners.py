import os
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

def build_banner(mode='dark'):
    avatar_path = '/Users/novice/Desktop/Github/RBCs-lang/assets/avatar.png'
    img = Image.open(avatar_path).convert('RGB')
    
    # 1. Crop head and shoulders
    w, h = img.size
    img = img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
    img = img.resize((60, 68), Image.Resampling.LANCZOS)
    
    # Contrast 1.3x + autocontrast + UnsharpMask
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    
    gray = np.array(img.convert('L'), dtype=float)
    h_g, w_g = gray.shape
    
    if mode == 'dark':
        mask = gray < 240
    else:
        mask = np.ones_like(gray, dtype=bool)
        
    # Floyd-Steinberg Error Diffusion (Serpentine order)
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

    portrait_color = "#A78BFA" if mode == 'dark' else "#7C3AED"
    chrome_color = "#22D3EE" if mode == 'dark' else "#0891B2"
    accent_color = "#10B981"
    bg_color = "#0A101F" if mode == 'dark' else "#F8FAFC"
    card_bg = "#0F172A" if mode == 'dark' else "#FFFFFF"
    text_muted = "#94A3B8" if mode == 'dark' else "#64748B"
    text_main = "#F8FAFC" if mode == 'dark' else "#0F172A"
    
    ox, oy = 50, 190
    dw, dh = 5, 5
    
    rect_elements = []
    for y in range(h_g):
        run_start = None
        for x in range(w_g):
            if mode == 'dark':
                draw_dot = (dithered[y, x] == 255) and mask[y, x]
            else:
                draw_dot = (dithered[y, x] == 0)
                
            if draw_dot:
                if run_start is None:
                    run_start = x
            else:
                if run_start is not None:
                    px = ox + run_start * dw
                    py = oy + y * dh
                    pw = (x - run_start) * dw
                    rect_elements.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{dh}" />')
                    run_start = None
        if run_start is not None:
            px = ox + run_start * dw
            py = oy + y * dh
            pw = (w_g - run_start) * dw
            rect_elements.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{dh}" />')
            
    portrait_rects_html = "\n    ".join(rect_elements)

    info_rows = [
        ("Subject", "Subh Sharma"),
        ("Role", "Frontend Engineer"),
        ("Origin", "Sonipat, Haryana, India"),
        ("Education", "B.Tech (CS, DS &amp; Business)"),
        ("Status", "Building + Learning + Shipping"),
        ("ToolChain", "VS Code · Git · Android Studio · Figma"),
        ("Core.Lang", "Python · JavaScript · HTML5 · CSS3"),
        ("Core.Frontend", "React · Next.js · Vite · Tailwind"),
        ("Core.Backend", "Node.js · Express · REST APIs"),
        ("Core.Database", "MongoDB · PostgreSQL · SQLite"),
        ("Core.Infra", "Git · GitHub · Vercel · Electron"),
        ("Grid.Mail", "Available on Request"),
        ("Grid.Portfolio", "Coming Soon"),
        ("Grid.LinkedIn", "linkedin.com/in/RBCs"),
        ("Grid.GitHub", "github.com/RBCs")
    ]
    
    text_elements = []
    y_start = 145
    line_spacing = 25
    
    for idx, (label, val) in enumerate(info_rows):
        y_pos = y_start + idx * line_spacing
        text_elements.append(f'''
        <g transform="translate(0, {y_pos})">
          <text x="440" y="0" fill="{text_muted}" font-family="Menlo, Monaco, 'Courier New', monospace" font-size="11.5" font-weight="500">{label}</text>
          <path d="M560 0 h420" stroke="{text_muted}" stroke-dasharray="2 4" stroke-width="1" opacity="0.3" transform="translate(0, -4)" />
          <text x="1120" y="0" text-anchor="end" fill="{text_main}" font-family="Menlo, Monaco, 'Courier New', monospace" font-size="11.5" font-weight="600" textLength="340" lengthAdjust="spacingAndGlyphs">{val}</text>
        </g>
        ''')

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">
  <style>
    .bg {{ fill: {bg_color}; }}
    .card {{ fill: {card_bg}; stroke: rgba(255,255,255,0.1); stroke-width: 1; }}
    .title-bar {{ fill: #1E293B; }}
    .dot-red {{ fill: #EF4444; }}
    .dot-yellow {{ fill: #F59E0B; }}
    .dot-green {{ fill: #10B981; }}
    .text-title {{ fill: {text_muted}; font-family: Menlo, Monaco, monospace; font-size: 13px; font-weight: 600; }}
    .live-badge {{ fill: #EF4444; }}
  </style>
  
  <rect width="1180" height="610" rx="16" class="bg" />
  
  <rect x="20" y="20" width="1140" height="570" rx="12" class="card" />
  <rect x="20" y="20" width="1140" height="42" rx="12" class="title-bar" />
  <rect x="20" y="50" width="1140" height="12" class="title-bar" />
  
  <circle cx="48" cy="41" r="6" class="dot-red" />
  <circle cx="68" cy="41" r="6" class="dot-yellow" />
  <circle cx="88" cy="41" r="6" class="dot-green" />
  <text x="590" y="45" text-anchor="middle" class="text-title">profile.sh --live</text>

  <g transform="translate(1010, 33)">
    <rect x="0" y="0" width="64" height="20" rx="10" fill="rgba(239, 68, 68, 0.15)" stroke="#EF4444" stroke-width="1"/>
    <circle cx="12" cy="10" r="4" class="dot-red" />
    <text x="24" y="14" fill="#EF4444" font-family="Menlo, Monaco, monospace" font-size="11" font-weight="700">LIVE</text>
  </g>
  <g transform="translate(910, 33)">
    <rect x="0" y="0" width="85" height="20" rx="10" fill="rgba(34, 211, 238, 0.15)" stroke="{chrome_color}" stroke-width="1"/>
    <text x="42.5" y="14" text-anchor="middle" fill="{chrome_color}" font-family="Menlo, Monaco, monospace" font-size="11" font-weight="700">@RBCs</text>
  </g>

  <rect x="40" y="85" width="360" height="480" rx="8" fill="rgba(15, 23, 42, 0.6)" stroke="{chrome_color}" stroke-width="1" opacity="0.8" />
  <text x="55" y="112" fill="{chrome_color}" font-family="Menlo, Monaco, monospace" font-size="12" font-weight="700" letter-spacing="1">VISUAL.MAP</text>
  
  <g fill="{portrait_color}" shape-rendering="crispEdges" transform="translate(0, 0)">
    {portrait_rects_html}
  </g>
  
  <text x="440" y="112" fill="{accent_color}" font-family="Menlo, Monaco, monospace" font-size="13" font-weight="700" letter-spacing="1">SYSTEM.INFO</text>
  <line x1="440" y1="122" x2="1120" y2="122" stroke="{accent_color}" stroke-width="1" opacity="0.3" />

  {"".join(text_elements)}

</svg>'''

    output_path = f'/Users/novice/Desktop/Github/RBCs-lang/assets/{mode}.svg'
    with open(output_path, 'w') as f:
        f.write(svg_content)
    print(f"Generated {output_path}")

build_banner('dark')
build_banner('light')
