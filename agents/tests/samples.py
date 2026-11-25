"""Valid outputs for the research schemas, as a model would return them."""

from typing import Any

NICHE: dict[str, Any] = {
    "summary": "Boutique hotels sell atmosphere and service; direct booking and local story win.",
    "characteristics": ["Under 40 rooms", "Owner-led", "Strong local identity"],
    "market_trends": ["Direct booking push", "Slow travel", "Wellness stays"],
    "opportunities": ["Packages", "Local partnerships", "Off-season events"],
    "challenges": ["OTA commissions", "Seasonality", "Small teams"],
    "typical_services": ["Breakfast", "Concierge", "Airport transfer"],
    "terminology": [
        {"term": "ADR", "meaning": "Average daily rate"},
        {"term": "RevPAR", "meaning": "Revenue per available room"},
        {"term": "Direct booking", "meaning": "Booking on the hotel's own site"},
        {"term": "Rate parity", "meaning": "Same price on every channel"},
    ],
}

SEGMENT = {"name": "Couples", "description": "Two to four night leisure stays"}
AUDIENCE: dict[str, Any] = {
    "summary": "Leisure couples and small groups who value design, calm and local character.",
    "primary_audiences": [SEGMENT],
    "secondary_audiences": [{"name": "Remote workers", "description": "Week-long stays"}],
    "demographics_and_interests": ["25-45", "Urban professionals", "Food and design"],
    "needs": ["Easy booking", "Clear photos", "Flexible cancellation"],
    "pain_points": ["Hidden fees", "Generic rooms", "Slow replies"],
    "motivations": ["Escape", "Celebration", "Discovery"],
    "buying_behavior": "Compare three sites, read reviews, book direct for perks.",
    "personas": [
        {
            "name": "Mira",
            "background": "Designer",
            "goals": ["Quiet weekend"],
            "frustrations": ["Noise"],
        },
        {
            "name": "Tom",
            "background": "Consultant",
            "goals": ["Work-friendly stay"],
            "frustrations": ["Bad wifi"],
        },
    ],
}

KW = {"term": "boutique hotel", "intent": "commercial"}
SEO: dict[str, Any] = {
    "summary": "Target boutique hotel and local-intent terms; build room, offers and story pages.",
    "primary_keywords": [KW, {"term": "book boutique hotel", "intent": "transactional"}, KW],
    "secondary_keywords": [KW, KW, KW],
    "long_tail_keywords": ["quiet boutique hotel with breakfast", "a", "b", "c", "d"],
    "recommended_site_structure": ["Home", "Rooms", "Offers"],
    "landing_pages": [
        {"page": "Rooms", "purpose": "Show rooms", "target_keyword": "boutique hotel rooms"},
        {"page": "Offers", "purpose": "Packages", "target_keyword": "hotel packages"},
    ],
    "content_topics": ["Local guide", "Seasonal events", "Our story"],
    "initial_strategy": "Rank for local commercial terms first, then grow with guides.",
}

REPORTS = {"niche": NICHE, "audience": AUDIENCE, "seo": SEO}

COMPETITOR: dict[str, Any] = {
    "summary": "Rivals lead with photos and a booking bar; story and local guides set one apart.",
    "competitors": [
        {
            "name": "Hotel Aster",
            "url": "https://hotelaster.example.com/",
            "positioning": "Quiet luxury in the old town.",
        },
        {
            "name": "The Lantern House",
            "url": "https://www.lanternhouse.example.org/stay",
            "positioning": "Design-led rooms with a rooftop bar.",
        },
    ],
    "common_site_structures": ["Home, Rooms, Offers", "Booking bar in the header", "Gallery"],
    "common_features": ["Online booking", "Room filters", "Newsletter sign-up"],
    "content_and_messaging": ["Calm, upscale tone", "Stress the location", "Few guest stories"],
    "differentiation_opportunities": ["Local guides", "Direct-booking perks", "Honest photos"],
}

BRAND: dict[str, Any] = {
    "summary": "Warm, calm boutique-hotel brand: deep green and gold, serif headings, local story.",
    "positioning": "A small owner-led hotel where design, calm and local character come first.",
    "personality_and_tone": {
        "traits": ["warm", "calm", "refined"],
        "voice": "Friendly and unhurried, like a host who knows the city well.",
        "do": ["Use short sentences", "Name real places"],
        "dont": ["Use hype words", "Shout in capitals"],
    },
    "messaging": {
        "tagline": "Stay like you live here",
        "value_proposition": "Book direct for the best rate, a late checkout and local tips.",
        "key_messages": ["Owner-led", "Book direct and save", "Rooms with a story"],
    },
    "palette": {
        "primary": "#1F4D3A",
        "secondary": "#C9A24B",
        "accent": "#8A5A44",
        "background": "#FAF7F0",
    },
    "typography": {
        "heading_font": "Playfair Display",
        "body_font": "Inter",
        "notes": "Playfair for headings and prices, Inter for everything else.",
    },
    "logo_direction": "A simple wordmark with a small cedar sprig, in the primary colour.",
    "image_guidelines": ["Natural light", "Real rooms, no stock", "Warm, muted tones"],
    "ui_ux_direction": ["Booking bar always visible", "Large photos", "Generous white space"],
    "inspiration": ["Aesop store pages", "Kinfolk magazine layouts"],
}
