"""`tokens.css`, written by code from the approved brand.

A model never writes it. The palette and the two fonts come from the guideline, and everything
the pages need beyond them (text colour, surfaces, the colour on a button) is worked out here from
the contrast of the colours, so text stays readable whatever palette was approved.
"""

from typing import Any

from webkiln_agents.workers.brand import contrast_ratio

DARK_TEXT = "#1a1a1a"
LIGHT_TEXT = "#ffffff"
FONTS_ORIGIN = "https://fonts.googleapis.com"

# Everything that does not come from the brand. Spacing steps are rems, so text size scales them.
STATIC_TOKENS = {
    "--space-1": "0.25rem",
    "--space-2": "0.5rem",
    "--space-3": "1rem",
    "--space-4": "1.5rem",
    "--space-5": "2rem",
    "--space-6": "3rem",
    "--space-7": "4.5rem",
    "--radius": "0.5rem",
    "--max-width": "68rem",
}


def _on(color: str) -> str:
    """Dark or light text, whichever reads better on `color`."""
    return max((DARK_TEXT, LIGHT_TEXT), key=lambda text: contrast_ratio(text, color))


def _mix(first: str, second: str, weight: float) -> str:
    """`first` with `weight` of `second` mixed in, both `#rrggbb`."""
    channels = (
        round(int(first[i : i + 2], 16) * (1 - weight) + int(second[i : i + 2], 16) * weight)
        for i in (1, 3, 5)
    )
    return "#" + "".join(f"{c:02x}" for c in channels)


def token_values(brand: dict[str, Any]) -> dict[str, str]:
    """The CSS variables of a brand guideline, by name."""
    palette = brand["palette"]
    typography = brand["typography"]
    background = palette["background"]
    text = _on(background)
    return {
        "--color-primary": palette["primary"],
        "--color-secondary": palette["secondary"],
        "--color-accent": palette["accent"],
        "--color-background": background,
        "--color-text": text,
        "--color-muted": _mix(text, background, 0.35),
        "--color-surface": _mix(background, text, 0.05),
        "--color-border": _mix(background, text, 0.15),
        "--color-on-primary": _on(palette["primary"]),
        "--color-on-secondary": _on(palette["secondary"]),
        "--color-on-accent": _on(palette["accent"]),
        "--font-heading": f'"{typography["heading_font"]}", system-ui, sans-serif',
        "--font-body": f'"{typography["body_font"]}", system-ui, sans-serif',
        **STATIC_TOKENS,
    }


def tokens_css(brand: dict[str, Any]) -> str:
    lines = [f"  {name}: {value};" for name, value in token_values(brand).items()]
    return (
        "/* Written by code from the approved brand guideline. Do not edit by hand. */\n:root {\n"
        + "\n".join(lines)
        + "\n}\n"
    )


def fonts_url(brand: dict[str, Any]) -> str:
    """The one external stylesheet: both fonts from Google Fonts. Names are letters, digits,
    spaces and hyphens only (see `FontName`), so they cannot break out of the URL."""
    typography = brand["typography"]
    names = dict.fromkeys((typography["heading_font"], typography["body_font"]))
    families = "&".join(f"family={name.replace(' ', '+')}:wght@400;600;700" for name in names)
    return f"{FONTS_ORIGIN}/css2?{families}&display=swap"
