# Standard
from typing import Dict, List, Optional
import json
import os

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import get_data_builder
from fms_dgt.base.task_card import TaskRunCard
from fms_dgt.index import DataBuilderIndex
import fms_dgt.utils as utils

sdg_logger = utils.sdg_logger


def generate_data(
    task_kwargs: Dict,
    builder_kwargs: Dict,
    data_paths: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    include_builder_paths: Optional[List[str]] = None,
    build_id: Optional[str] = None,
):
    """Generate data for a set of tasks using their respective data builders

    Args:
        task_kwargs (Dict): A dictionary of keyword arguments to pass to each task.
        builder_kwargs (Dict): A dictionary of keyword arguments to pass to each data builder.
        data_paths (Optional[List[str]], optional): A list of paths to data files.
        config_path (Optional[str], optional): A path to a configuration file.
        include_builder_paths (Optional[List[str]], optional): A list of paths to search for data builders.
        build_id (Optional[str], optional): An ID to associate with all of the tasks executed in this run.
    """
    data_paths = data_paths or []
    builder_overrides = None
    task_overrides = dict()

    assert (
        len(task_kwargs) == 1
    ), f"Cannot use 'db_composition' databuilder with multiple arguments"

    task_files = task_kwargs[0].pop(_TASK_FILES_KEY)
    task_overrides = task_kwargs[0].pop(_TASK_KEY, dict())
    dependency_graph = task_kwargs[0].pop(_DEP_GRAPH)

    task_kwargs[0].pop("task_card")

    if task_files is None:
        raise ValueError(f"'{_TASK_FILES_KEY}' field must be present in task config")
    if dependency_graph is None:
        raise ValueError(f"'{_DEP_GRAPH}' field must be present in task config")

    # gather task initializations from task_files
    task_inits = []
    for data_path in task_files:
        if data_path and os.path.exists(data_path):
            for task_init in utils.read_data(data_path):
                task_init = {
                    **task_kwargs[0],
                    **task_overrides.get(task_init["task_name"], dict()),
                    **task_init,
                    "save_formatted_output": True,
                }
                task_inits.append(task_init)
        else:
            raise FileExistsError(f"Error: data path ({data_path}) does not exist.")

    # construct dependency graph
    task_names = [task_init["task_name"] for task_init in task_inits]
    for k, v in dependency_graph.items():
        if type(v) != list:
            raise ValueError(
                f"All values of '{_DEP_GRAPH}' field must be provided as lists"
            )
        for el in [k] + v:
            if el not in task_names:
                raise ValueError(f"Missing task file for {el}")

    builder_index = DataBuilderIndex()

    builder_list = [t["data_builder"] for t in self._task_inits]
    builder_names = self._builder_index.match_builders(builder_list)
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

    builder_cfgs = list(self._builder_index.load_builder_configs(builder_names).items())

    datastore_map = dict()

    groups = comp_utils.extract_exec_sets(self._dependencies)
    for group in groups:

        task_group = [t for t in self._task_inits if t["task_name"] in group]

        group_final_datastore = None
        group_input_datastore = None

        for builder_name, builder_cfg in builder_cfgs:
            builder_info = self._builder_index.builder_index[builder_name]
            builder_dir = builder_info.get("builder_dir")
            if isinstance(builder_cfg, tuple):
                _, builder_cfg = builder_cfg
                if builder_cfg is None:
                    continue

            sdg_logger.debug("Builder config for %s: %s", builder_name, builder_cfg)

            all_builder_kwargs = {
                "config": builder_cfg,
                "task_kwargs": [
                    {
                        # get task card
                        "task_card": TaskRunCard(
                            task_name=task_init.get("task_name"),
                            databuilder_name=task_init.get("data_builder"),
                            task_spec=json.dumps(task_init),
                            databuilder_spec=json.dumps(
                                utils.load_nested_paths(builder_cfg, builder_dir)
                            ),
                        ),
                        # other params
                        **task_init,
                    }
                    for task_init in task_group
                    if task_init["data_builder"] == builder_name
                ],
                **self._builder_kwargs,
            }

            try:
                # first see if databuilder is loaded by default
                data_builder: DataBuilder = get_data_builder(
                    builder_name, **all_builder_kwargs
                )
            except KeyError as e:
                if f"Attempted to load data builder '{builder_name}'" in str(e):
                    utils.import_builder(builder_dir)
                    data_builder: DataBuilder = get_data_builder(
                        builder_name, **all_builder_kwargs
                    )
                else:
                    raise e

            ### CORE LOGIC
            # NOTE: this is the CORE part of this databuilder
            # assign
            for task in data_builder._tasks:
                # construct new datastore
                if task.name in self._dependencies:
                    for subtask in self._dependencies[task.name]:
                        group_input_datastore = datastore_map[subtask]

            for task in data_builder._tasks:
                if group_final_datastore is None:
                    group_final_datastore = task._final_datastore
                task._final_datastore = group_final_datastore
                datastore_map[task.name] = group_final_datastore
                if group_input_datastore is not None:
                    task._init_dataloader(group_input_datastore)

            # TODO: ship this off
            data_builder.execute_tasks()
