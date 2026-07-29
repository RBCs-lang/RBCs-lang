import os
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

def create_logo_masks():
    # Generate 3 logo point sets (Python, JS, Git) inside a 160x160 box centered in VISUAL.MAP (cx=220, cy=320)
    size = 160
    
    # 1. Python Logo (Two interlocking snake shapes)
    py_img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(py_img)
    # Outer boundaries / python shield / snakes
    draw.rounded_rectangle([30, 20, 130, 75], radius=20, outline=255, width=18)
    draw.rounded_rectangle([30, 85, 130, 140], radius=20, outline=255, width=18)
    draw.ellipse([45, 32, 60, 47], fill=255) # Eye 1
    draw.ellipse([100, 113, 115, 128], fill=255) # Eye 2
    py_pts = np.argwhere(np.array(py_img) > 128) # (y, x)
    
    # 2. JS Logo (Square boundary with JS text)
    js_img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(js_img)
    draw.rectangle([20, 20, 140, 140], outline=255, width=12)
    # J
    draw.line([(70, 70), (70, 115), (50, 115), (50, 100)], fill=255, width=12)
    # S
    draw.line([(120, 75), (95, 75), (95, 93), (120, 93), (120, 115), (95, 115)], fill=255, width=12)
    js_pts = np.argwhere(np.array(js_img) > 128)
    
    # 3. Git Logo (Diamond with branch lines)
    git_img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(git_img)
    # Rotated square / diamond outline
    draw.polygon([(80, 15), (145, 80), (80, 145), (15, 80)], outline=255, width=12)
    # Branch lines & nodes
    draw.line([(55, 80), (105, 80)], fill=255, width=10)
    draw.line([(80, 55), (80, 105)], fill=255, width=10)
    draw.line([(80, 80), (110, 110)], fill=255, width=10)
    draw.ellipse([50, 72, 66, 88], fill=255)
    draw.ellipse([100, 72, 116, 88], fill=255)
    draw.ellipse([102, 102, 118, 118], fill=255)
    git_pts = np.argwhere(np.array(git_img) > 128)
    
    return py_pts, js_pts, git_pts

def sample_pts(pts, target_n=900):
    if len(pts) == 0:
        return np.zeros((target_n, 2))
    indices = np.random.choice(len(pts), target_n, replace=(len(pts) < target_n))
    return pts[indices]

def generate_animated_svg(mode='dark'):
    avatar_path = '/Users/novice/Desktop/Github/RBCs-lang/assets/avatar.png'
    img = Image.open(avatar_path).convert('RGB')
    
    # 1. Crop & resize head/shoulders
    w, h = img.size
    img = img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
    img = img.resize((75, 85), Image.Resampling.LANCZOS)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    
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

    portrait_color = "#A78BFA" if mode == 'dark' else "#7C3AED"
    chrome_color = "#22D3EE" if mode == 'dark' else "#0891B2"
    accent_color = "#10B981"
    bg_color = "#0A101F" if mode == 'dark' else "#F8FAFC"
    card_bg = "#0F172A" if mode == 'dark' else "#FFFFFF"
    text_muted = "#94A3B8" if mode == 'dark' else "#64748B"
    text_main = "#F8FAFC" if mode == 'dark' else "#0F172A"
    
    ox, oy = 50, 190
    dw, dh = 4, 4
    
    # 2. Build Portrait Layer Dots (~1500 dots) with per-dot noise & band grouping for SMIL drift
    portrait_dots = []
    for y in range(h_g):
        for x in range(w_g):
            if (dithered[y, x] == 255 and mask[y, x]) if mode == 'dark' else (dithered[y, x] == 0):
                px = ox + x * dw
                py = oy + y * dh
                portrait_dots.append((px, py))
                
    # Group portrait dots into 94 bands for the morphing drift loop
    np.random.seed(42)
    n_dots = len(portrait_dots)
    bands = 94
    dot_indices = np.arange(n_dots)
    np.random.shuffle(dot_indices)
    band_groups = np.array_split(dot_indices, bands)
    
    portrait_paths_html = []
    for b_idx, group in enumerate(band_groups):
        if len(group) == 0: continue
        # Band offset noise
        dx = int(np.random.normal(0, 18))
        dy = int(np.random.normal(0, 18))
        
        path_data = []
        for idx in group:
            px, py = portrait_dots[idx]
            path_data.append(f"M{px},{py}h{dw}v{dh}h-{dw}z")
            
        d_str = " ".join(path_data)
        # SMIL loop animation (17.4s total loop)
        # keyTimes: 0 (portrait hold 3s), 0.23 (fade/drift to logo 1), 0.40 (logo 1 hold), 0.57 (logo 2 hold), 0.74 (logo 3 hold), 1.0 (return)
        animate_transform = f'''
        <animateTransform
          attributeName="transform"
          type="translate"
          values="0,0; {dx},{dy}; 0,0; {dx},{dy}; 0,0"
          keyTimes="0; 0.25; 0.50; 0.75; 1"
          dur="17.4s"
          repeatCount="indefinite" />
        <animate
          attributeName="opacity"
          values="1; 0.15; 1; 0.15; 1"
          keyTimes="0; 0.25; 0.50; 0.75; 1"
          dur="17.4s"
          repeatCount="indefinite" />
        '''
        portrait_paths_html.append(f'<g><path d="{d_str}" fill="{portrait_color}" shape-rendering="crispEdges" />{animate_transform}</g>')
        
    # 3. Travellers Layer: 260 dots morphing between Python, JS, and Git logos
    py_pts, js_pts, git_pts = create_logo_masks()
    py_sampled = sample_pts(py_pts, 260)
    js_sampled = sample_pts(js_pts, 260)
    git_sampled = sample_pts(git_pts, 260)
    
    # Center logos in VISUAL.MAP frame (cx=220, cy=320)
    # Box is 160x160, so origin is (220 - 80) = 140, (320 - 80) = 240
    logo_ox, logo_oy = 140, 240
    
    traveller_rects = []
    for i in range(260):
        # Coordinates for Python, JS, Git
        py_x = logo_ox + py_sampled[i, 1]
        py_y = logo_oy + py_sampled[i, 0]
        
        js_x = logo_ox + js_sampled[i, 1]
        js_y = logo_oy + js_sampled[i, 0]
        
        git_x = logo_ox + git_sampled[i, 1]
        git_y = logo_oy + git_sampled[i, 0]
        
        # SMIL keyTimes: 0 (hidden), 0.20 (fade in Python), 0.45 (morph to JS), 0.70 (morph to Git), 1.0 (fade out return to portrait)
        anim_x = f'<animate attributeName="x" values="{py_x:.1f}; {py_x:.1f}; {js_x:.1f}; {git_x:.1f}; {py_x:.1f}" keyTimes="0; 0.25; 0.50; 0.75; 1" dur="17.4s" repeatCount="indefinite"/>'
        anim_y = f'<animate attributeName="y" values="{py_y:.1f}; {py_y:.1f}; {js_y:.1f}; {git_y:.1f}; {py_y:.1f}" keyTimes="0; 0.25; 0.50; 0.75; 1" dur="17.4s" repeatCount="indefinite"/>'
        anim_op = f'<animate attributeName="opacity" values="0; 0.9; 0.9; 0.9; 0" keyTimes="0; 0.20; 0.50; 0.75; 1" dur="17.4s" repeatCount="indefinite"/>'
        
        traveller_rects.append(f'<rect x="{py_x:.1f}" y="{py_y:.1f}" width="2.5" height="2.5" fill="{chrome_color}">{anim_x}{anim_y}{anim_op}</rect>')
        
    travellers_html = "\n    ".join(traveller_rects)

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
  
  <!-- Layer 1: Portrait Drift Bands -->
  {"".join(portrait_paths_html)}
  
  <!-- Layer 2: Vector Travellers Morphing Between Python, JS, and Git Logos -->
  <g>
    {travellers_html}
  </g>
  
  <text x="440" y="112" fill="{accent_color}" font-family="Menlo, Monaco, monospace" font-size="13" font-weight="700" letter-spacing="1">SYSTEM.INFO</text>
  <line x1="440" y1="122" x2="1120" y2="122" stroke="{accent_color}" stroke-width="1" opacity="0.3" />

  {"".join(text_elements)}

</svg>'''

    output_path = f'/Users/novice/Desktop/Github/RBCs-lang/assets/{mode}.svg'
    with open(output_path, 'w') as f:
        f.write(svg_content)
    print(f"Generated {output_path}")

generate_animated_svg('dark')
generate_animated_svg('light')
