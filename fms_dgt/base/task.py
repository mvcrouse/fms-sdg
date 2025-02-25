# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union
import dataclasses
import os
import random

# Local
from fms_dgt.base.datastore import BaseDatastore, DatastoreDataType
from fms_dgt.base.registry import get_dataloader, get_datastore
from fms_dgt.base.task_card import TaskRunCard
from fms_dgt.constants import TYPE_KEY
from fms_dgt.utils import group_data_by_attribute, init_dataclass_from_dict

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MACHINE_BATCH_SIZE = 10
DEFAULT_SEED_BATCH_SIZE = 100
DEFAULT_NUM_OUTPUTS = 2

###
# Runner args
###


@dataclass
class TaskRunnerConfig:
    """Configuration for an SDG task runner, i.e., specifies parameters guiding how SDG is run

    Attributes:
        output_dir (Optional[str]): The directory where the generated outputs will be saved.
        save_formatted_output (Optional[bool]): A boolean indicating whether to save outputs that have been reformatted
        restart_generation (Optional[bool]): A boolean indicating whether to restart generation from scratch.
        seed_batch_size (Optional[int]): The batch size used for seed examples.
        machine_batch_size (Optional[int]): The batch size used for machine examples.
        num_outputs_to_generate (Optional[int]): The number of outputs to generate.
    """

    output_dir: Optional[str] = None
    save_formatted_output: Optional[bool] = False
    restart_generation: Optional[bool] = False
    seed_batch_size: Optional[int] = None
    machine_batch_size: Optional[int] = None
    num_outputs_to_generate: Optional[int] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = DEFAULT_OUTPUT_DIR
        if self.seed_batch_size is None:
            self.seed_batch_size = DEFAULT_SEED_BATCH_SIZE
        if self.machine_batch_size is None:
            self.machine_batch_size = DEFAULT_MACHINE_BATCH_SIZE
        if self.num_outputs_to_generate is None:
            self.num_outputs_to_generate = DEFAULT_NUM_OUTPUTS


###
# Data class
###


@dataclass
class SdgData:
    """This class is intended to hold the seed / machine generated instruction data"""

    task_name: str

    def to_dict(self) -> Dict:
        """Returns output dictionary representation of dataclass. Designed to be overridden with custom logic.

        Returns:
            Dict: Dictionary representation of dataclass
        """
        return asdict(self)

    @classmethod
    def get_field_names(cls):
        return [field.name for field in dataclasses.fields(cls)]


###
# Main task class
###


class SdgTask:
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = SdgData
    OUTPUT_DATA_TYPE: SdgData = None

    def __init__(
        self,
        task_name: str,
        task_description: str,
        created_by: str,
        data_builder: str,
        task_card: TaskRunCard,
        runner_config: Union[Mapping, TaskRunnerConfig],
        instruction_format: Optional[Dict[str, str]] = None,
        datastore: Optional[Dict] = None,
        seed_datastore: Optional[Dict] = None,
        final_datastore: Optional[Dict] = None,
        dataloader: Optional[Dict] = None,
        seed_examples: Optional[List[Any]] = None,
        store_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initializes task object.

        Args:
            config (Union[Mapping, DataBuilderConfig], optional): Config specifying all databuilder settings.
            task_name (str): The name of the Task object.
            task_description (str): A description of the SDG task is designed to solve.
            created_by (str): The name of the individual / group who created the code assistant.
            data_builder (str): The name of the data builder that should be used to process this task.
            task_card (TaskCard): The task card containing all experiment information.
            runner_config (Union[Mapping, TaskRunnerConfig], optional): Config specifying the run settings of the task.
            instruction_format (Optional[Dict[str, str]]): A dictionary template that can be used to translate intermediate data objects to instruction-tuning pairs.
            datastore (Optional[Dict]): A dictionary containing the configuration for the datastore.
            seed_datastore (Optional[Dict]): A dictionary containing the configuration for the seed datastore.
            final_datastore (Optional[Dict]): A dictionary containing the configuration for the datastore used for storing final data.
            dataloader (Optional[Dict]): A dictionary containing the configuration for the dataloader.
            seed_examples (Optional[List[Any]]): A list of seed examples.
            store_name (Optional[str]): A base name to use for the datastores. Will be set to [task_name] if None

        """
        if self.OUTPUT_DATA_TYPE is None:
            self.OUTPUT_DATA_TYPE = self.INPUT_DATA_TYPE

        self._name = task_name
        self._task_description = task_description
        self._created_by = created_by
        self._data_builder = data_builder
        self._instruction_format = instruction_format
        self._dataloader = dataloader
        self._seed_examples = seed_examples

        self._store_name = (
            store_name or (datastore or dict()).pop("store_name", None) or self._name
        )
        self._kwargs = kwargs
        self._runner_config = init_dataclass_from_dict(runner_config, TaskRunnerConfig)

        self._task_card = task_card
        self._output_dir = self._runner_config.output_dir
        self._save_formatted_output = self._runner_config.save_formatted_output
        self._restart_generation = self._runner_config.restart_generation
        self._post_proc_id = 0

        self.machine_data = []

        self._seed_batch_size = self._runner_config.seed_batch_size
        self._machine_batch_size = self._runner_config.machine_batch_size
        self._num_outputs_to_generate = self._runner_config.num_outputs_to_generate

        for attr in [
            "seed_batch_size",
            "machine_batch_size",
            "num_outputs_to_generate",
        ]:
            if getattr(self, f"_{attr}") < 0:
                raise ValueError(
                    f"Cannot have negative value of {getattr(self, f'_{attr}')} for {attr} parameter"
                )

        # dataloader params
        self._dataloader_cfg = (
            dataloader if dataloader is not None else {TYPE_KEY: "default"}
        )

        # datastore params
        base_store_cfg = {
            "restart": self.restart_generation,
            "output_dir": self._output_dir,
            "task_card": self.task_card,
        }
        self._datastore_cfg = {
            **base_store_cfg,
            **(datastore if datastore is not None else {TYPE_KEY: "default"}),
        }
        self._seed_datastore_cfg = {
            **base_store_cfg,
            **(seed_datastore if seed_datastore is not None else {TYPE_KEY: "default"}),
        }
        self._final_datastore_cfg = {
            **base_store_cfg,
            **(final_datastore if final_datastore is not None else self._datastore_cfg),
        }
        self._task_card_datastore_cfg = {**base_store_cfg, **self._datastore_cfg}

        self._dataloader_state_datastore: BaseDatastore = None
        self._datastore: BaseDatastore = None
        self._final_datastore: BaseDatastore = None

        self._save_task_card()
        self._init_dataloader()
        self._init_datastores()

    @property
    def runner_config(self) -> TaskRunnerConfig:
        """Returns the run config of the task.

        Returns:
            TaskRunnerConfig: Run config for the task
        """
        return self._runner_config

    @property
    def name(self) -> str:
        """Returns name of task.

        Returns:
            str: Name of task
        """
        return self._name

    @property
    def task_description(self) -> str:
        """Returns the task description.

        Returns:
            str: Task description
        """
        return self._task_description

    @property
    def restart_generation(self):
        return self._restart_generation

    @property
    def task_card(self) -> TaskRunCard:
        """Returns the task card.

        Returns:
            TaskRunCard: Task card
        """
        return self._task_card

    @property
    def datastore(self) -> BaseDatastore:
        """Returns the datastore of the class.

        Returns:
            BaseDatastore: Datastore
        """
        return self._datastore

    @property
    def final_datastore(self) -> BaseDatastore:
        """Returns the datastore of the class.

        Returns:
            BaseDatastore: Datastore
        """
        return self._final_datastore

    def _save_task_card(self):
        """Saves experiment card to datastore."""

        exp_ds_kwargs = {
            "store_name": os.path.join(self._store_name, "task_card"),
            "data_type": DatastoreDataType.CARD,
            **self._task_card_datastore_cfg,
        }
        task_card_datastore = get_datastore(
            exp_ds_kwargs.get(TYPE_KEY), **exp_ds_kwargs
        )

        prev_card = None
        if not self._restart_generation:
            prev_task_cards: List[Dict] = [
                card
                for card in task_card_datastore.load_data()
                if card["build_id"] == self.task_card.build_id
            ]
            if prev_task_cards:
                prev_card = TaskRunCard(**prev_task_cards[-1])
                self.task_card.run_id = prev_card.run_id

        assert (
            self.task_card.run_id is not None
        ), "TaskCard.run_id cannot be set to None"

        task_card_datastore.save_data([self.task_card.to_dict()])
        task_card_datastore.close()

    def _init_dataloader(self) -> None:
        """Initialize datastore object for storing all SDG data."""

        # init seed datastore for dataloader
        seed_ds_kwargs = {
            "store_name": os.path.join(self._store_name, "seed_data"),
            "data": self._seed_examples,
            "data_type": DatastoreDataType.SEED,
            **self._seed_datastore_cfg,
            "restart": False,
        }
        seed_datastore = get_datastore(
            self._seed_datastore_cfg.get(TYPE_KEY), **seed_ds_kwargs
        )

        # init dataloader state datastore (should be same as base datastore)
        dls_ds_kwargs = {
            "store_name": os.path.join(self._store_name, "dataloader_state"),
            "data_type": DatastoreDataType.STATE,
            **self._datastore_cfg,
        }
        self._dataloader_state_datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY), **dls_ds_kwargs
        )

        self._dataloader_state: Any = None

        # init dataloader itself
        self._dataloader = get_dataloader(
            self._dataloader_cfg.get(TYPE_KEY),
            data=seed_datastore.load_data(),
            **self._dataloader_cfg,
        )

    def _init_datastores(self):

        # init input/output datastore
        io_ds_kwargs = {
            "store_name": os.path.join(self._store_name, "data"),
            "data_type": DatastoreDataType.TASK_DATA,
            **self._datastore_cfg,
        }
        self._datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY), **io_ds_kwargs
        )

        # init final output datastore (should be same as input/output datastore)
        final_ds_kwargs = {
            "store_name": os.path.join(self._store_name, "final_data"),
            "data_type": DatastoreDataType.FINAL_DATA,
            **self._final_datastore_cfg,
            "restart": True,  # always restart final datastore
        }
        self._final_datastore = get_datastore(
            self._final_datastore_cfg.get(TYPE_KEY), **final_ds_kwargs
        )

    def set_new_datastore(self):
        """Sets default datastore (which is used to gather data for final_datastore)

        Args:
            datastore (BaseDatastore): Datastore to set
        """
        self._post_proc_id += 1
        pp_ds_kwargs = {
            "store_name": os.path.join(
                self._store_name, f"postproc_data_{self._post_proc_id}"
            ),
            "data_type": DatastoreDataType.TASK_DATA,
            **self._datastore_cfg,
            "restart": True,
        }

        # close existing datastore before updating
        self._datastore.close()

        # update pointer to new datastore
        self._datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY), **pp_ds_kwargs
        )

    def instantiate_input_example(self, **kwargs: Any) -> INPUT_DATA_TYPE:
        """Instantiate an input example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an input example object.

        Returns:
            INPUT_DATA_TYPE: An instance of INPUT_DATA_TYPE.
        """
        return self.INPUT_DATA_TYPE(
            task_name=kwargs.pop("task_name", self.name), **kwargs
        )

    def instantiate_output_example(self, **kwargs: Any) -> OUTPUT_DATA_TYPE:  # type: ignore
        """Instantiate an output example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an output example object.

        Returns:
            OUTPUT_DATA_TYPE: An instance of OUTPUT_DATA_TYPE.
        """
        return self.OUTPUT_DATA_TYPE(**kwargs)

    def instantiate_instruction(self, data: OUTPUT_DATA_TYPE) -> Dict:  # type: ignore
        """Instantiates an instruction-tuning pair from output data instance.

        Args:
            data (OUTPUT_DATA_TYPE): Data to be converted to instruction-tuning pair.

        Returns:
            Dict: Dictionary representing an instruction-tuning pair.
        """

        assert (
            self._instruction_format is not None
        ), f"'instruction_format' cannot be None in method 'instantiate_instruction'"

        data = data if type(data) == dict else asdict(data)
        output = dict(self._instruction_format)
        for k in output.keys():
            for ds_k, ds_v in data.items():
                inp_key = "{{" + ds_k + "}}"
                if inp_key in output[k]:
                    output[k] = output[k].replace(inp_key, str(ds_v))

        return output

    def get_example(self) -> SdgData:
        """Returns single example from dataloader.

        Returns:
            SdgData: Example to be used for SDG.
        """
        try:
            return self.instantiate_input_example(**next(self._dataloader))
        except StopIteration:
            return None

    def get_batch_examples(self) -> List[SdgData]:
        """Returns batch of examples from dataloader. Mixes examples from seed data and machine-generated data.

        Returns:
            List[SdgData]: List of examples to be used by SDG process.
        """
        outputs = []

        # get outputs from seed data loader sequentially
        for _ in range(self._seed_batch_size):
            example = self.get_example()
            if example is None:
                break
            outputs.append(example)

        # get outputs from machine batch randomly
        m_data = self.machine_data
        if m_data and len(m_data) > self._machine_batch_size:
            m_data = random.sample(m_data, k=self._machine_batch_size)

        outputs.extend(m_data)

        return outputs

    def is_complete(self) -> bool:
        """Indicates whether SDG task has completed.

        Returns:
            bool: Whether task is complete or not.
        """
        return len(self.machine_data) > self._num_outputs_to_generate

    def save_intermediate_data(
        self,
        new_data: Union[SdgData, List[SdgData]],
    ) -> None:
        """Saves intermediate data produced during SDG (useful for checkpointing).

        Args:
            new_data (Union[SdgData, List[SdgData]]): List of SdgData to save.
        """
        if type(new_data) != list:
            new_data: List[SdgData] = [new_data]

        to_save = [d if type(d) == dict else d.to_dict() for d in new_data]
        self.datastore.save_data(to_save)

    def load_intermediate_data(self) -> List[SdgData]:
        """Loads intermediate data produced during SDG (will be used to resume SDG). This function loads the data from datastore, which is either
            the latest datastore defined during post processing or the original input/output datastore.

        Returns:
            List[SdgData]: List of SdgData that has been loaded
        """
        loaded_data = self.datastore.load_data()
        if loaded_data:
            self.machine_data = [
                self.instantiate_output_example(**d) for d in loaded_data
            ]

    def save_final_data(self) -> None:
        """Saves final instruction-tuning data that can be used directly for training."""
        if self._save_formatted_output:
            loaded_data = self.datastore.load_data() or []
            to_add = [
                self.instantiate_instruction(self.instantiate_output_example(**d))
                for d in loaded_data
            ]
            if to_add:
                self._final_datastore.save_data(to_add)

    def load_final_data(self) -> List[SdgData]:
        """Loads final data produced during SDG (will be used to resume SDG).

        Returns:
            List[SdgData]: List of SdgData that has been loaded
        """
        return self._final_datastore.load_data()

    def save_dataloader_state(self):
        curr_state = self._dataloader.get_state()
        if self._dataloader_state != curr_state:
            self._dataloader_state = curr_state
            self._dataloader_state_datastore.save_data([{"state": curr_state}])

    def load_dataloader_state(self):
        prev_state = self._dataloader_state_datastore.load_data()
        if prev_state:
            self._dataloader.set_state(prev_state[-1]["state"])
            self._dataloader_state = prev_state

    def finish(self) -> None:
        """Method for wrapping up task execution. Called after `is_complete` signals task has completed"""
        # close datastores, which may involve writing any buffered data
        self._dataloader_state_datastore.close()
        self.datastore.close()

        # save final data
        self.save_final_data()
        self._final_datastore.close()


###
# Transformation task class
###


@dataclass
class TransformTaskRunnerConfig(TaskRunnerConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.seed_batch_size is None:
            self.seed_batch_size = 100
        if self.machine_batch_size is None:
            self.machine_batch_size = 0


class TransformTask(SdgTask):
    """TransformTask is a subclass of SdgTask that has default values that are more conducive to transformation tasks."""

    def __init__(
        self,
        *args,
        runner_config: Union[Mapping, TaskRunnerConfig],
        dataloader: Optional[Dict] = None,
        **kwargs,
    ):
        runner_config = init_dataclass_from_dict(
            runner_config, TransformTaskRunnerConfig
        )
        # adjust dataloader to not loop
        if dataloader is None:
            dataloader = {TYPE_KEY: "default", "loop_over_data": False}

        super().__init__(
            *args,
            runner_config=runner_config,
            dataloader=dataloader,
            **kwargs,
        )


###
# Utilities
###

T = TypeVar("T")


def group_data_by_task(data_list: List[T]) -> List[List[T]]:
    """Utility function that groups input data by task name.

    Args:
        data_list (List[T]): List of SdgData to group into tasks

    Returns:
        List[List[T]]: SdgData that has been grouped into tasks
    """
    return group_data_by_attribute(data_list, "task_name")
