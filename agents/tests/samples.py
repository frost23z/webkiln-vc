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
