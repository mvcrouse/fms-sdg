# Standard
from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Union
import json
import time

# Third Party
from ray.actor import ActorHandle
from tqdm import tqdm
import ray

# Local
from fms_dgt.base.block import get_row_name
from fms_dgt.base.databuilder import DataBuilder, TransformationDataBuilder
from fms_dgt.base.registry import get_data_builder_class
from fms_dgt.base.task import SdgData, SdgTask, TransformTask
from fms_dgt.blocks.postprocessors import BasePostProcessingBlock
from fms_dgt.utils import sdg_logger

DEFAULT_MAX_STALLED_ATTEMPTS = 5
DEFAULT_MAX_GEN_REQUESTS = 10
DEFAULT_NUM_WORKERS = 1


class DataBuilderOrchestrator(ABC):
    """A data builder head collects the execution results for each task constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    WORKER_CLASS: DataBuilder = DataBuilder

    def __init__(
        self,
        builder_name: str,
        task_kwargs: List[dict],
        max_gen_requests: int = DEFAULT_MAX_GEN_REQUESTS,
        max_stalled_requests: int = DEFAULT_MAX_STALLED_ATTEMPTS,
        parallel_workers: int = DEFAULT_NUM_WORKERS,
        **databuilder_kwargs: dict,
    ) -> None:
        """Initializes data builder object.

        Args:
            builder_name (str): Name of databuilder.
            task_kwargs (List[dict]): List of task_kwargs for each task to be executed by this data builder.

        Kwargs:
            max_gen_requests (int, optional): Maximum number of data generation loop iterations to execute before terminating.
            max_stalled_requests (int, optional): Maximum number of data generation loop iterations that do not return new data before terminating.
            parallel_workers (int, optional): Number of workers to distribute SDG generation tasks to.
            databuilder_kwargs (List[dict]): Dictionary of databuilder_kwargs needed to initialize data builder.
        """
        self._name = builder_name

        self._databuilder_kwargs = databuilder_kwargs
        self._task_kwargs = task_kwargs

        self._max_gen_requests = max_gen_requests
        self._max_stalled_requests = max_stalled_requests
        self._parallel_workers = parallel_workers

        self._db_class: DataBuilder = get_data_builder_class(builder_name)

        # initialize tasks
        self._tasks: List[SdgTask] = None
        self._init_tasks()

        # initialize ray workers
        self._workers: List[ActorHandle] = None
        self._init_workers()

    @property
    def name(self) -> str:
        """Returns the name of the data builder

        Returns:
            str: name string
        """
        return self._name

    @property
    def tasks(self) -> List[SdgTask]:
        """Returns the tasks associated with this class.

        Returns:
            List[SdgTask]: List of tasks to be used in this data builder
        """
        return self._tasks

    def _init_tasks(self):
        """Initializes the tasks for this data builder"""
        self._tasks: List[SdgTask] = [
            self._db_class.TASK_TYPE(**task_kwargs) for task_kwargs in self._task_kwargs
        ]

    def _init_workers(self):
        """Initializes the ray actors that will be in charge of producing data"""

        self._workers: List[ActorHandle] = []

        build_id = self._tasks[0].task_card.build_id if self._tasks else None

        # TODO: relax assumption that we're working with one node
        node = ray.nodes()[0]
        worker_cpus = int(node["Resources"].get("CPU", 1) // self._parallel_workers)
        worker_gpus = int(node["Resources"].get("GPU", 0) // self._parallel_workers)

        for _ in range(self._parallel_workers):
            actor = ray.remote(num_cpus=worker_cpus, num_gpus=worker_gpus)(
                self._db_class
            ).remote(
                **{
                    "task_kwargs": self._task_kwargs,
                    "build_id": build_id,
                    **self._databuilder_kwargs,
                },
            )
            self._workers.append(actor)

    def execute_tasks(self):
        """Main entry point for task execution. Default behavior executes a loop until all tasks are complete, where each loop generates synthetic data."""

        # load the LM-generated data
        for task in self._tasks:
            task.load_intermediate_data()
            if task.machine_data:
                sdg_logger.debug(
                    "Loaded %s machine-generated data", len(task.machine_data)
                )
            task.load_dataloader_state()

        # main entry point to task execution
        generating = [task for task in self._tasks if not task.is_complete()]
        completed = [task for task in self._tasks if task.is_complete()]

        generate_start = time.time()

        stalled_cts = {task.name: self._max_stalled_requests for task in generating}

        request_idx = 0
        # outer loop captures postprocessing
        while generating and request_idx <= self._max_gen_requests:
            # inner loop captures main generation
            progress_bar = tqdm(total=len(generating), desc="Running generation tasks")
            postprocessing: List[SdgTask] = []
            while generating and request_idx <= self._max_gen_requests:

                request_idx += 1

                filtered_data: List[SdgData] = []
                for generated_inst in self.call_with_task_list(generating):
                    # save incrementally
                    task = next(
                        task
                        for task in generating
                        if get_row_name(generated_inst) == task.name
                    )
                    task.save_intermediate_data(generated_inst)
                    filtered_data.append(generated_inst)
                    task.save_dataloader_state()

                for task in generating:
                    new_data = [
                        gen_inst
                        for gen_inst in filtered_data
                        if get_row_name(gen_inst) == task.name
                    ]
                    task.machine_data.extend(new_data)

                    stalled_cts[task.name] -= 1
                    if new_data:
                        stalled_cts[task.name] = self._max_stalled_requests

                    if task.is_complete() or stalled_cts[task.name] <= 0:
                        postprocessing.append(task)
                        progress_bar.update()

                # remove tasks from generating that have completed
                generating = [task for task in generating if task not in postprocessing]

                sdg_logger.info(
                    "Generated %s data in this iteration, %s data overall",
                    len(filtered_data),
                    sum(
                        [
                            len(task.machine_data)
                            for task in (generating + postprocessing + completed)
                        ]
                    ),
                )

            # launch postprocessing for completed tasks
            sdg_logger.info("Launch postprocessing")
            self.execute_postprocessing(postprocessing)
            sdg_logger.info("Postprocessing completed")

            for task in postprocessing:
                if task.is_complete() or stalled_cts[task.name] <= 0:
                    if stalled_cts[task.name] <= 0:
                        sdg_logger.info(
                            "Task %s has not produced any data in the last %s attempts, terminating task",
                            task.name,
                            self._max_stalled_requests,
                        )
                    completed.append(task)
                    task.finish()

            # redefine generating and postprocessing
            generating = [task for task in postprocessing if task not in completed]

            progress_bar.close()

        generate_duration = time.time() - generate_start
        sdg_logger.info("Generation took %.2fs", generate_duration)

    def call_with_task_list(self, tasks: List[SdgTask]) -> Iterable[SdgData]:
        """Executes data builder __call__ function for all in-progress tasks. Is executed in the inner loop of `execute_tasks`

        Args:
            request_idx (int): The iteration of `execute_tasks` this method was called at
            tasks (List[SdgTask]): List of in-progress tasks

        Returns:
            Iterable[SdgData]: List of data instances generated by the __call__ function
        """
        actor_results = []
        for actor in self._workers:
            data_pool = [task.get_builder_inputs() for task in tasks]
            actor_results.append(actor.__call__.remote(data_pool))
        generated_data = [d for gen_data in ray.get(actor_results) for d in gen_data]
        return generated_data

    def execute_postprocessing(self, completed_tasks: List[SdgTask]):
        """Executes any postprocessing required after tasks have completed.

        Args:
            completed_tasks (List[SdgTask]): tasks that have been completed and can undergo postprocessing
        """

        # TODO: fix
        return

        post_proc_blocks = [
            b for b in self.blocks if isinstance(b, BasePostProcessingBlock)
        ]
        if post_proc_blocks:
            datastore_assgns = {
                task.name: [task.datastore, task.make_postprocess_datastore()]
                for task in completed_tasks
            }
            for i, block in enumerate(post_proc_blocks, start=1):
                block_inputs = [
                    (
                        task.name,
                        *datastore_assgns[task.name],
                    )
                    for task in completed_tasks
                ]
                # execute postprocessing
                block.generate(block_inputs)
                # update datastores
                datastore_assgns = {
                    task.name: [
                        datastore_assgns[task.name][-1],
                        task.make_postprocess_datastore(),
                    ]
                    for task in completed_tasks
                }
            for task in completed_tasks:
                task.set_postprocess_datastore(datastore_assgns[task.name][-1])
                # load_intermediate_data loads from postprocess datastore
                task.load_intermediate_data()


###
# Transformation-specific databuilder
###


class TransformationDataBuilderOrchestrator(DataBuilderOrchestrator):
    """This class is designed to have sensible default methods for transformation use cases"""

    WORKER_CLASS: DataBuilder = TransformationDataBuilder

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        for attr in ["get_builder_inputs"]:
            if getattr(SdgTask, attr) != getattr(self._db_class.TASK_TYPE, attr):
                raise ValueError(
                    f"Subclasses of TransformTask, cannot override [{attr}] in subclass [{self._db_class.TASK_TYPE}]"
                )

    def execute_tasks(self):
        """Main entry point for task execution. Default behavior iterates over all tasks and applies the transformation to each task's data."""
        tasks = self._tasks + []

        for task in tasks:
            assert isinstance(
                task, TransformTask
            ), f"Task {task.name} must inherit from TransformTask class to be used with TransformationDataBuilder"
            task.load_dataloader_state()

        progress_bar = tqdm(total=len(tasks), desc="Running transformation tasks")
        generate_start = time.time()

        filtered_data: List[SdgData] = []
        for generated_inst in self.call_with_task_list(tasks):
            # save incrementally
            task = next(
                task for task in tasks if get_row_name(generated_inst) == task.name
            )
            task.save_intermediate_data(generated_inst)
            filtered_data.append(generated_inst)
            task.save_dataloader_state()

        for task in tasks:
            new_data = [
                gen_inst
                for gen_inst in filtered_data
                if get_row_name(generated_inst) == task.name
            ]
            task.machine_data.extend(new_data)
            progress_bar.update()

        sdg_logger.info(
            "Generated %s data",
            sum([len(task.machine_data) for task in tasks]),
        )

        progress_bar.close()

        sdg_logger.info("Launch postprocessing")
        self.execute_postprocessing(tasks)
        sdg_logger.info("Postprocessing completed")

        for task in tasks:
            task.finish()

        generate_duration = time.time() - generate_start
        sdg_logger.info("Generation took %.2fs", generate_duration)

    def call_with_task_list(self, tasks: List[SdgTask]) -> Iterable[SdgData]:
        """Executes data builder __call__ function for all in-progress tasks.

        Args:
            tasks (List[SdgTask]): List of in-progress tasks

        Returns:
            Iterable[SdgData]: List of data instances generated by the __call__ function
        """
        # default behavior is to simply extract the seed / machine generated data and pass to data builder

        data_pool = [e for task in tasks for e in task.get_builder_inputs()]
        while data_pool:
            partition_size = len(data_pool) // len(self._workers)
            actor_results = [
                self._workers[i].__call__.remote(
                    data_pool[i * partition_size : (i + 1) * partition_size]
                )
                for i in range(len(self._workers))
            ]
            for gen_data in ray.get(actor_results):
                for d in gen_data:
                    yield d

            data_pool = [e for task in tasks for e in task.get_builder_inputs()]
