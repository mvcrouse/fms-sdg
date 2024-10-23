# Standard
from typing import Dict, List, Optional, Type
import json
import os

# Third Party
import ray

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.orchestrator import DataBuilderOrchestrator
from fms_dgt.base.registry import get_data_builder_class
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
    parallel_workers: Optional[int] = None,
):
    """Generate data for a set of tasks using their respective data builders

    Args:
        task_kwargs (Dict): A dictionary of keyword arguments to pass to each task.
        builder_kwargs (Dict): A dictionary of keyword arguments to pass to each data builder.
        data_paths (Optional[List[str]], optional): A list of paths to data files.
        config_path (Optional[str], optional): A path to a configuration file.
        include_builder_paths (Optional[List[str]], optional): A list of paths to search for data builders.
        build_id (Optional[str], optional): An ID to associate with all of the tasks executed in this run.
        parallel_workers (Optional[int], optional): Parallelize databuilder generation up to N workers
    """
    data_paths = data_paths or []
    builder_overrides = None
    task_overrides = dict()
    if config_path:
        (
            addlt_data_paths,
            builder_overrides,
            task_overrides,
        ) = utils.load_joint_config(config_path)
        data_paths.extend(addlt_data_paths)

    if not data_paths and not config_path:
        raise ValueError(
            f"One of ['data-paths', 'config-path'] must be provided in the arguments"
        )

    data_paths = list(set(data_paths))

    # check data_path first then seed_tasks_path
    # throw an error if both not found
    # pylint: disable=broad-exception-caught,raise-missing-from
    task_inits = []
    for data_path in data_paths:
        if data_path and os.path.exists(data_path):
            for task_init in utils.read_data(data_path):
                task_init = {
                    **task_overrides.get(task_init["task_name"], dict()),
                    **task_init,
                }
                task_inits.append(task_init)
        else:
            raise FileExistsError(f"Error: data path ({data_path}) does not exist.")

    # gather data builders here
    builder_list = [t["data_builder"] for t in task_inits]
    builder_index = DataBuilderIndex(
        include_builder_paths=include_builder_paths,
    )
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

    # load builder configs
    all_builder_cfgs = list(
        builder_index.load_builder_configs(
            builder_names, config_overrides=builder_overrides
        ).items()
    )

    # TODO: data
    """ray.init(address: str | None = None, *, num_cpus: int | None = None, num_gpus: int | None = None, resources: Dict[str, float] | None = None, labels: Dict[str, str] | None = None, object_store_memory: int | None = None, local_mode: bool = False, ignore_reinit_error: bool = False, include_dashboard: bool | None = None, dashboard_host: str = '127.0.0.1', dashboard_port: int | None = None, job_config: ray.job_config.JobConfig = None, configure_logging: bool = True, logging_level: int = 'info', logging_format: str | None = None, logging_config: LoggingConfig | None = None, log_to_driver: bool | None = None, namespace: str | None = None, runtime_env: Dict[str, Any] | RuntimeEnv | None = None, storage: str | None = None, **kwargs)"""
    ray.init()

    for builder_name, builder_cfg in all_builder_cfgs:

        # we batch together tasks at the level of data builders
        builder_info = builder_index.builder_index[builder_name]
        builder_dir = builder_info.get("builder_dir")
        if isinstance(builder_cfg, tuple):
            _, builder_cfg = builder_cfg
            if builder_cfg is None:
                continue

        sdg_logger.debug("Builder config for %s: %s", builder_name, builder_cfg)

        all_builder_params = {
            "builder_name": builder_name,
            "task_kwargs": [
                {
                    # get task card
                    "task_card": TaskRunCard(
                        task_name=task_init.get("task_name"),
                        databuilder_name=task_init.get("data_builder"),
                        task_spec=json.dumps({**task_init, **task_kwargs}),
                        databuilder_spec=json.dumps(
                            utils.load_nested_paths(builder_cfg, builder_dir)
                        ),
                        build_id=build_id,
                    ),
                    # other params
                    **{**task_init, **task_kwargs},
                }
                for task_init in task_inits
                if task_init["data_builder"] == builder_name
            ],
            "config": builder_cfg,
            **builder_kwargs,
        }

        # no nesting of errors
        err = None
        try:
            # first see if databuilder is loaded by default
            orchestrator_class = get_orchestrator_class(builder_name)
            data_builder_orchestrator = orchestrator_class(**all_builder_params)
        except KeyError as e:
            err = e

        if f"Attempted to load data builder '{builder_name}'" in str(err):
            utils.import_builder(builder_dir)
            orchestrator_class = get_orchestrator_class(builder_name)
            data_builder_orchestrator = orchestrator_class(**all_builder_params)
        else:
            raise err

        # TODO: ship this off
        data_builder_orchestrator.execute_tasks()

        # TODO: cleanup
        del data_builder_orchestrator


def get_orchestrator_class(builder_name: str) -> Type[DataBuilderOrchestrator]:
    def _get_all_subclasses(
        cls: DataBuilderOrchestrator,
    ) -> List[DataBuilderOrchestrator]:
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.extend(_get_all_subclasses(subclass))
        return all_subclasses + [cls]

    db_class: DataBuilder = get_data_builder_class(builder_name)
    known_subclasses = _get_all_subclasses(DataBuilderOrchestrator)

    for orchestrator_class in known_subclasses:
        if not getattr(orchestrator_class, "WORKER_CLASS"):
            raise ValueError(
                f"When defining a DataBuilderOrchestrator, the [WORKER_CLASS] field cannot be null"
            )
        if issubclass(db_class, orchestrator_class.WORKER_CLASS):
            return orchestrator_class

    raise ValueError(
        f"Unexpected data builder type [{db_class}] for builder [{builder_name}]"
    )
