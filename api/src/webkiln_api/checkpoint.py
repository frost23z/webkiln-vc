from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ALLOWED_TYPES = [
    ("webkiln_agents.core.schemas", "Input"),
]

serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)
