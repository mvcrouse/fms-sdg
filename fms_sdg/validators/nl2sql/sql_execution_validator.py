# Standard
from typing import Any, Dict, List
import logging
import sqlite3

# Third Party
import sqlglot

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_validator
from fms_sdg.base.validator import BaseValidator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@register_validator("sql_execution_validator")
class SQLExecutionValidator(BaseValidator):
    """SQL execution validator."""

    def validate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        """Validate a batch.

        Args:
            inputs: list of instances.
        """
        for x in inputs:
            x.result = self._validate(*x.args, **x.kwargs)

    def _validate(self, record: Dict[str, str], **kwargs: Any) -> bool:
        """Validate a record containing information on schema, query and utterance.

        Args:
            record: record containing the generated sample information.
        """
        is_valid = False
        try:
            connection = sqlite3.connect(":memory:")
            # NOTE: make sure the schema is compatible with SQLite
            parsed_schema = sqlglot.parse(record["sql_schema"])
            sqlite_schema = ";\n".join(
                [
                    parsed_element.sql(dialect="sqlite")
                    for parsed_element in parsed_schema
                    if parsed_element
                ]
            )
            # NOTE: here we might have multiple statements to define a schema
            cursor = connection.executescript(sqlite_schema)
            # TODO: consider enabling insertion of random records in tables
            # NOTE: make sure the query is compatible with SQLite
            parsed_query = sqlglot.parse_one(record["sql_query"])
            if parsed_query:
                sqlite_query = parsed_query.sql(dialect="sqlite")
                cursor.execute(sqlite_query)
                # TODO: consider using the result of the fetch
                # for further validation in the future
                _ = cursor.fetchall()
                is_valid = True
        except Exception:
            logger.warning(f"discarded generated record={record}")
        return is_valid
