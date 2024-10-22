# Standard
from typing import Dict, Iterable, List, Mapping, Type, Union

# Third Party
from ray.actor import ActorHandle
from ray.experimental.state.api import list_actors
import ray

# Local
from fms_dgt.base.databuilder import DataBuilder, DataBuilderConfig
from fms_dgt.base.registry import get_block_class, get_data_builder_class
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.blocks.postprocessors import BasePostProcessingBlock
from fms_dgt.constants import BLOCKS_KEY, NAME_KEY
from fms_dgt.utils import sdg_logger


class ParallelDataBuilder(DataBuilder):
    """A data builder represents a means of constructing data for a set of tasks"""

    def __init__(
        self,
        builder_name: str,
        parallel_workers: int,
        *args,
        config: Union[Mapping, DataBuilderConfig] = None,
        task_kwargs: List[dict] = None,
        **kwargs,
    ) -> None:
        """Initializes data builder object.

        Args:
            builder_name (str): Databuilder to parallelize
            parallel_workers (int): Number of workers to parallelize across
            config (dict): Contains all settings that would be passed to the databuilder
        """

        self._actors: List[ActorHandle] = []

        # validate databuilder can be parallelized
        db_class = get_data_builder_class(builder_name)
        if db_class:
            for attr in ["execute_tasks", "call_with_task_list"]:
                if getattr(DataBuilder, attr) != getattr(db_class, attr):
                    raise ValueError(
                        f"Method [{attr}] cannot be defined in class {db_class} if --parallel-workers is used"
                    )

        # TODO: relax assumption that we're working with one node
        node = ray.nodes()[0]
        worker_cpus = int(node["Resources"]["CPU"] // parallel_workers)
        worker_gpus = int(node["Resources"]["GPU"] // parallel_workers)

        for _ in range(parallel_workers):
            actor = ray.remote(num_cpus=worker_cpus, num_gpus=worker_gpus)(
                db_class
            ).remote(
                *args,
                **{
                    "config": config,
                    "task_kwargs": [{**tk, "is_worker": True} for tk in task_kwargs],
                    **kwargs,
                },
            )
            self._actors.append(actor)

        # only initialize postprocessing blocks for the synchronization process
        config[BLOCKS_KEY] = [
            block
            for block in config[BLOCKS_KEY]
            if isinstance(get_block_class(block[NAME_KEY], BasePostProcessingBlock))
        ]
        self._db = self._make_dynamic_subclass(
            db_class, *args, **{"config": config, "task_kwargs": task_kwargs, **kwargs}
        )

    def _make_dynamic_subclass(
        self, db_class: Type[DataBuilder], *args: List, **kwargs: Dict
    ):
        """Creates a new object with the explicit purpose of overriding [call_with_task_list] to distribute generation across multiple Ray actors

        Args:
            db_class (Type[DataBuilder]): Databuilder type to make subclass of
            args (List): Args to initialize databuilder
            kwargs (Dict): Kwargs to initialize databuilder
        """

        # dynamically create subclass of db_class
        class ParallelDataBuilderSubclass(db_class):
            def __init__(self, actors: List[ActorHandle], *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._actors = actors

            def call_with_task_list(
                self, request_idx: int, tasks: List[SdgTask]
            ) -> Iterable[SdgData]:
                actor_results = []
                for actor in self._actors:
                    data_pool = [e for task in tasks for e in task.get_batch_examples()]
                    args = [request_idx, data_pool]
                    kwargs = dict()
                    actor_results.append(actor.__call__.remote(*args, **kwargs))
                generated_data = [
                    d for gen_data in ray.get(actor_results) for d in gen_data
                ]
                return generated_data

        return ParallelDataBuilderSubclass(self._actors, *args, **kwargs)

    def execute_tasks(self):
        return self._db.execute_tasks()
