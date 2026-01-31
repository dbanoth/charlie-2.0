# Chatbot Customization Guide

This document explains how to customize the chatbot's branding, including the name and logo.

## Changing the Chatbot Name

To change the chatbot name (e.g., from "Livestock Advisor" to "Charlie"), edit the following locations in `api.py`:

| Line | Description | Current Value |
|------|-------------|---------------|
| 53   | Startup log message | `"Starting Livestock Advisor API..."` |
| 61   | Shutdown log message | `"Shutting down Livestock Advisor API..."` |
| 65   | FastAPI title | `title="Livestock Advisor API"` |
| 141  | HTML page title | `<title>Livestock Advisor</title>` |
| 284  | Header title | `<h1>Livestock Advisor</h1>` |
| 289  | Welcome message | `<h2>Welcome to Livestock Advisor</h2>` |
| 321  | Welcome message (JS) | `<h2>Welcome to Livestock Advisor</h2>` |

## Changing the Icon/Logo

### Option 1: Using an Emoji (Current Setup)

The current icon is set via CSS at **line 177** in `api.py`:

```css
.header h1::before { content: "🐄"; }
```

Simply replace `🐄` with any emoji you prefer.

### Option 2: Using a Custom Logo Image

#### Step 1: Set Up Static File Serving

Add the following import at the top of `api.py`:

```python
from fastapi.staticfiles import StaticFiles
```

After the `app = FastAPI(...)` block, add:

```python
app.mount("/static", StaticFiles(directory="static"), name="static")
```

#### Step 2: Create the Static Directory

Create a `static` folder in the project root:

```
hitl-agriculture-agent/
├── static/
│   └── logo.png    # Your logo file
├── api.py
└── ...
```

#### Step 3: Update the CSS (Line 177)

Replace the emoji with a background image:

```css
.header h1::before {
    content: "";
    display: inline-block;
    width: 24px;
    height: 24px;
    background: url('/static/logo.png') no-repeat center/contain;
    margin-right: 0.5rem;
}
```

#### Alternative: Use an IMG Tag

Instead of CSS, you can add an `<img>` tag directly in the HTML.

1. Remove or comment out line 177 (the `::before` CSS)

2. Update line 284 to include the image:

```html
<h1><img src="/static/logo.png" alt="Logo" style="height: 24px; vertical-align: middle; margin-right: 0.5rem;"> Charlie</h1>
```

## Logo Recommendations

- **Format**: PNG with transparent background or SVG
- **Size**: 24x24px to 48x48px for the header
- **Location**: Store in the `static/` directory

## Quick Reference

| Customization | File | Line(s) |
|---------------|------|---------|
| App name | api.py | 53, 61, 65, 141, 284, 289, 321 |
| Header icon | api.py | 177 |
| Welcome text | api.py | 289-290, 321-322 |
| Suggestion buttons | api.py | 292-294 |
