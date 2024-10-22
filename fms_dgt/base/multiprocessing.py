# Standard
from typing import Dict, Iterable, List, Mapping, Type, Union

# Third Party
from ray.actor import ActorHandle
import ray

# Local
from fms_dgt.base.databuilder import DataBuilder, DataBuilderConfig
from fms_dgt.base.registry import get_data_builder_class
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.utils import sdg_logger


class ParallelDataBuilder(DataBuilder):
    """A data builder represents a means of constructing data for a set of tasks"""

    def __init__(
        self,
        builder_name: str,
        *args,
        config: Union[Mapping, DataBuilderConfig] = None,
        **kwargs,
    ) -> None:
        """Initializes data builder object.

        Args:
            builder_name (str): Databuilder to parallelize
            config (dict): Contains all settings that would be passed to the databuilder.
        """

        self._actors: List[ActorHandle] = []

        # validate databuilder can be parallelized
        db_class = get_data_builder_class(builder_name)
        if db_class:
            for attr in ["execute_tasks", "call_with_task_list"]:
                if getattr(DataBuilder, attr) != getattr(db_class, attr):
                    raise ValueError(
                        f"Method [{attr}] cannot be defined in class {db_class} if --parallelize flag is used"
                    )

        for _ in range(2):
            actor = ray.remote(num_cpus=1, num_gpus=1)(db_class).remote(
                *args, **{"config": config, **kwargs}
            )
            self._actors.append(actor)

        # we don't want to initialize blocks for the synchronization process
        config["blocks"] = []
        self._db = self._make_dynamic_subclass(
            db_class, *args, **{"config": config, **kwargs}
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
