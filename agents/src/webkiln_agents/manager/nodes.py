from webkiln_agents.core.schemas import Input


def process_initial_input(state: Input) -> dict:
    return {"initial_input": state}
