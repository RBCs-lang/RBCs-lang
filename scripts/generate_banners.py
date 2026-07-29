import os
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

def create_crisp_icon_pts(target_n=600):
    size = 260
    
    # 1. Python Logo (Filled iconic dual snakes)
    py_img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(py_img)
    # Upper Snake
    draw.rounded_rectangle([30, 20, 230, 120], radius=40, fill=255)
    draw.rectangle([130, 70, 230, 120], fill=0) # Cutout right bottom
    draw.rounded_rectangle([130, 70, 230, 120], radius=20, fill=255)
    draw.ellipse([70, 50, 95, 75], fill=0) # Eye 1
    # Lower Snake
    draw.rounded_rectangle([30, 140, 230, 240], radius=40, fill=255)
    draw.rectangle([30, 140, 130, 190], fill=0) # Cutout left top
    draw.rounded_rectangle([30, 140, 130, 190], radius=20, fill=255)
    draw.ellipse([165, 185, 190, 210], fill=0) # Eye 2
    
    # 2. JS Logo (Filled badge with clear JS cutout)
    js_img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(js_img)
    draw.rounded_rectangle([20, 20, 240, 240], radius=20, fill=255)
    # J cutout
    draw.line([(140, 100), (140, 195), (95, 195), (95, 160)], fill=0, width=22)
    # S cutout
    draw.line([(215, 110), (165, 110), (165, 145), (215, 145), (215, 195), (165, 195)], fill=0, width=22)
    
    # 3. Git Logo (Filled diamond with branch lines & nodes)
    git_img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(git_img)
    draw.polygon([(130, 15), (245, 130), (130, 245), (15, 130)], fill=255)
    # Internal branch paths cutouts / nodes
    draw.line([(80, 130), (180, 130)], fill=0, width=16)
    draw.line([(130, 80), (130, 180)], fill=0, width=16)
    draw.line([(130, 130), (190, 190)], fill=0, width=16)
    draw.ellipse([62, 112, 98, 148], fill=255)
    draw.ellipse([162, 112, 198, 148], fill=255)
    draw.ellipse([172, 172, 208, 208], fill=255)
    
    py_pts = np.argwhere(np.array(py_img) > 128)
    js_pts = np.argwhere(np.array(js_img) > 128)
    git_pts = np.argwhere(np.array(git_img) > 128)
    
    def sample_exact(pts, n):
        idx = np.random.choice(len(pts), n, replace=(len(pts) < n))
        return pts[idx]
        
    return sample_exact(py_pts, target_n), sample_exact(js_pts, target_n), sample_exact(git_pts, target_n)

def build_hd_morphing_banner(mode='dark'):
    avatar_path = '/Users/novice/Desktop/Github/RBCs-lang/assets/avatar.png'
    img = Image.open(avatar_path).convert('RGB')
    
    # Crop head and shoulders
    w, h = img.size
    img = img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
    img = img.resize((70, 80), Image.Resampling.LANCZOS)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.35)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=150))
    
    gray = np.array(img.convert('L'), dtype=float)
    h_g, w_g = gray.shape
    
    mask = (gray < 240) if mode == 'dark' else np.ones_like(gray, dtype=bool)
        
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

    ox, oy = 50, 150
    dw, dh = 4, 4
    
    portrait_pts = []
    for y in range(h_g):
        for x in range(w_g):
            if (dithered[y, x] == 255 and mask[y, x]) if mode == 'dark' else (dithered[y, x] == 0):
                px = ox + x * dw
                py = oy + y * dh
                portrait_pts.append((px, py))
                
    N_DOTS = 600
    np.random.seed(42)
    p_indices = np.random.choice(len(portrait_pts), N_DOTS, replace=(len(portrait_pts) < N_DOTS))
    P0 = np.array([portrait_pts[i] for i in p_indices])
    
    # 260x260 Logo canvas perfectly centered in VISUAL.MAP (ox=90, oy=170)
    logo_ox, logo_oy = 90, 170
    py_raw, js_raw, git_raw = create_crisp_icon_pts(N_DOTS)
    
    # Map (y, x) -> (X, Y)
    P1 = np.column_stack([logo_ox + py_raw[:, 1], logo_oy + py_raw[:, 0]])  # Python
    P2 = np.column_stack([logo_ox + js_raw[:, 1], logo_oy + js_raw[:, 0]])  # JS
    P3 = np.column_stack([logo_ox + git_raw[:, 1], logo_oy + git_raw[:, 0]]) # Git
    
    # Optimal Transport Matching using Hungarian algorithm
    cost_01 = np.linalg.norm(P0[:, None, :] - P1[None, :, :], axis=2)
    _, match_01 = linear_sum_assignment(cost_01)
    P1_matched = P1[match_01]
    
    cost_12 = np.linalg.norm(P1_matched[:, None, :] - P2[None, :, :], axis=2)
    _, match_12 = linear_sum_assignment(cost_12)
    P2_matched = P2[match_12]
    
    cost_23 = np.linalg.norm(P2_matched[:, None, :] - P3[None, :, :], axis=2)
    _, match_23 = linear_sum_assignment(cost_23)
    P3_matched = P3[match_23]
    
    portrait_color = "#A78BFA" if mode == 'dark' else "#7C3AED"
    chrome_color = "#22D3EE" if mode == 'dark' else "#0891B2"
    accent_color = "#10B981"
    bg_color = "#0A101F" if mode == 'dark' else "#F8FAFC"
    card_bg = "#0F172A" if mode == 'dark' else "#FFFFFF"
    text_muted = "#94A3B8" if mode == 'dark' else "#64748B"
    text_main = "#F8FAFC" if mode == 'dark' else "#0F172A"
    
    # Total duration: 16 Seconds
    key_times = "0; 0.1563; 0.25; 0.4063; 0.50; 0.6563; 0.75; 0.9063; 1"
    
    morph_elements = []
    for i in range(N_DOTS):
        x0, y0 = int(P0[i, 0]), int(P0[i, 1])
        x1, y1 = int(P1_matched[i, 0]), int(P1_matched[i, 1])
        x2, y2 = int(P2_matched[i, 0]), int(P2_matched[i, 1])
        x3, y3 = int(P3_matched[i, 0]), int(P3_matched[i, 1])
        
        vals_x = f"{x0};{x0};{x1};{x1};{x2};{x2};{x3};{x3};{x0}"
        vals_y = f"{y0};{y0};{y1};{y1};{y2};{y2};{y3};{y3};{y0}"
        
        anim_x = f'<animate attributeName="x" values="{vals_x}" keyTimes="{key_times}" dur="16s" repeatCount="indefinite"/>'
        anim_y = f'<animate attributeName="y" values="{vals_y}" keyTimes="{key_times}" dur="16s" repeatCount="indefinite"/>'
        
        morph_elements.append(f'<rect x="{x0}" y="{y0}" width="3.5" height="3.5" rx="0.8" fill="{portrait_color}">{anim_x}{anim_y}</rect>')
        
    morph_html = "\n    ".join(morph_elements)

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

  <!-- Left Frame: VISUAL.MAP -->
  <rect x="40" y="85" width="360" height="480" rx="8" fill="rgba(15, 23, 42, 0.6)" stroke="{chrome_color}" stroke-width="1" opacity="0.8" />
  <text x="55" y="112" fill="{chrome_color}" font-family="Menlo, Monaco, monospace" font-size="12" font-weight="700" letter-spacing="1">VISUAL.MAP</text>
  
  <!-- Filled Iconic Particle Layer (600 Particles) -->
  <g shape-rendering="crispEdges">
    {morph_html}
  </g>
  
  <text x="440" y="112" fill="{accent_color}" font-family="Menlo, Monaco, monospace" font-size="13" font-weight="700" letter-spacing="1">SYSTEM.INFO</text>
  <line x1="440" y1="122" x2="1120" y2="122" stroke="{accent_color}" stroke-width="1" opacity="0.3" />

  {"".join(text_elements)}

</svg>'''

    output_path = f'/Users/novice/Desktop/Github/RBCs-lang/assets/{mode}.svg'
    with open(output_path, 'w') as f:
        f.write(svg_content)
    print(f"Generated {output_path}")

build_hd_morphing_banner('dark')
build_hd_morphing_banner('light')
