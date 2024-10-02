# Standard
from typing import Any, List, Mapping
import os

# Local
from fms_dgt import utils
from fms_dgt.base.databuilder import DataBuilder, DataBuilderConfig
from fms_dgt.base.registry import register_data_builder
from fms_dgt.databuilders.nonstandard.compositional import utils as comp_utils
from fms_dgt.index import DataBuilderIndex
from fms_dgt.utils import sdg_logger

_TASK_FILES_KEY = "task_files"
_TASK_KEY = "tasks"
_DEP_GRAPH = "dependencies"


@register_data_builder("db_composition")
class CompositionalDataBuilder(DataBuilder):
    """Class for compositional databuilder task"""

    def __init__(
        self,
        config: Mapping | DataBuilderConfig = None,
        task_kwargs: List[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, task_kwargs, **kwargs)

        assert (
            len(task_kwargs) == 1
        ), f"Cannot use 'db_composition' databuilder with multiple arguments"

        task_files = task_kwargs[0].get(_TASK_FILES_KEY)
        task_overrides = task_kwargs[0].get(_TASK_KEY, dict())
        dependency_graph = task_kwargs[0].get(_DEP_GRAPH)

        if task_files is None:
            raise ValueError(
                f"'{_TASK_FILES_KEY}' field must be present in task config"
            )
        if dependency_graph is None:
            raise ValueError(f"'{_DEP_GRAPH}' field must be present in task config")

        # gather task initializations from task_files
        task_inits = []
        for data_path in task_files:
            if data_path and os.path.exists(data_path):
                for task_init in utils.read_data(data_path):
                    task_init = {
                        **task_overrides.get(task_init["task_name"], dict()),
                        **task_init,
                    }
                    task_inits.append(task_init)
            else:
                raise FileExistsError(f"Error: data path ({data_path}) does not exist.")

        # construct dependency graph
        task_names = [task_init["task_name"] for task_init in task_inits]
        for k, v in dependency_graph:
            if type(v) != list:
                raise ValueError(
                    f"All values of '{_DEP_GRAPH}' field must be provided as lists"
                )
            for el in [k] + v:
                if el not in task_names:
                    raise ValueError(f"Missing task file for {el}")
        dag_ordering = comp_utils.extract_dag(dependency_graph)

        builder_list = [t["data_builder"] for t in task_inits]
        builder_index = DataBuilderIndex()
        builder_names = builder_index.match_builders(builder_list)
        sdg_logger.debug("All builders: %s", builder_names)
        builder_missing = set(
            [
                builder
                for builder in builder_list
                if builder not in builder_names and "*" not in builder
            ]
        )

        if builder_missing:
            missing = ", ".join(builder_missing)
            raise ValueError(f"Builder specifications not found: [{missing}]")

    def execute_tasks(self):
        pass
