from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine
from webkiln_api.db import sqlalchemy_url
from webkiln_api.tables import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def include_object(
    object_: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    """Leave tables that are not in our metadata alone.

    The LangGraph checkpointer creates and migrates its own tables (`checkpoints`, `checkpoint_*`)
    in the same database. Without this filter, autogenerate sees them as unknown and writes
    `drop_table` for them.
    """
    return not (type_ == "table" and reflected and compare_to is None)


def run_migrations_online() -> None:
    # Migrations run in a plain synchronous connection, on the URL the API itself uses.
    engine = create_engine(sqlalchemy_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_object=include_object
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
