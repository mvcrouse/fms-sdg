# Standard
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union
import json
import time

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.block import BaseBlock, get_row_name
from fms_dgt.base.registry import get_block
from fms_dgt.base.task import SdgData, SdgTask, TransformTask
from fms_dgt.base.task_card import TaskRunCard
from fms_dgt.blocks.generators.llm import CachingLM
from fms_dgt.blocks.postprocessors import BasePostProcessingBlock
from fms_dgt.constants import NAME_KEY, TYPE_KEY
from fms_dgt.utils import all_annotations, init_dataclass_from_dict, sdg_logger

DEFAULT_MAX_STALLED_ATTEMPTS = 5
DEFAULT_MAX_GEN_REQUESTS = 10


@dataclass
class DataBuilderConfig(dict):
    """Configuration for a data builder.

    Attributes:
        name (Optional[str]): The name of the data builder.
        blocks (Optional[List[Dict]]): A list of block configurations.
        metadata (Optional[Dict[str, Any]]): Metadata for the data builder. Allows for users to pass arbitrary info to data builders
    """

    name: Optional[str] = None
    blocks: Optional[dict] = None
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.blocks is None:
            self.blocks = []


class DataBuilder(ABC):
    """A data builder represents a means of constructing data for a set of tasks"""

    TASK_TYPE: SdgTask = SdgTask

    def __init__(
        self,
        config: Union[Mapping, DataBuilderConfig] = None,
        build_id: str = None,
        **kwargs: Dict,
    ):
        """Initializes data builder object.

        Args:
            config (Union[Mapping, DataBuilderConfig], optional): Config specifying all databuilder settings.
            build_id (str, optional): ID to specify particular run of generation.
            kwargs (Dict, optional): Any additional kwargs for the databuilder
        """
        self._config = init_dataclass_from_dict(config, DataBuilderConfig)
        self._build_id = build_id
        self._kwargs = kwargs

        # initialize blocks
        self._init_blocks()

    @property
    def name(self) -> str:
        """Returns the name of the data builder

        Returns:
            str: name string
        """
        return self.config.name

    @property
    def config(self) -> DataBuilderConfig:
        """Returns the DataBuilderConfig associated with this class.

        Returns:
            DataBuilderConfig: Config specifying data builder settings
        """
        return self._config

    @property
    def blocks(self) -> List[BaseBlock]:
        """Returns the blocks associated with this class.

        Returns:
            List[BaseBlock]: List of blocks to be used in this data builder
        """
        return self._blocks

    def _init_blocks(self):
        """This method does two things:

        (1) It initializes each block object specified in self.config.blocks
        (2) It sets the block-attributes for a DataBuilder to be those initialized blocks (where the block is assumed to be assigned to `obj_name`)
            - In the process of doing this, it checks that the type specified in the DataBuilder class's attribute matches the block type that was initialized

        This method is intended to be overloaded when type checking is not necessary (e.g., in the case of the Pipeline class).
        """
        self._blocks: List[BaseBlock] = []

        # TODO: need to handle nested blocks
        for obj_kwargs in self.config.blocks:

            for req_key in (NAME_KEY, TYPE_KEY):
                assert (
                    req_key in obj_kwargs
                ), f"'{req_key}' field missing in data builder config from block with args:\n{json.dumps(obj_kwargs, indent=4)} "

            obj_name = obj_kwargs.get("name")
            obj_type = obj_kwargs.get(TYPE_KEY)

            assert not any(
                block.name == obj_name for block in self._blocks
            ), f"Duplicate '{obj_name}' block in '{self.name}' data builder"

            obj = get_block(
                obj_type, build_id=self._build_id, builder_name=self.name, **obj_kwargs
            )

            # we type check when not using a pipeline
            type_annotations = all_annotations(type(self))
            assert (
                obj_name in type_annotations
            ), f"Object {obj_name} is missing from definition of DataBuilder {self.__class__}"

            obj_type = type_annotations[obj_name]

            # double check types
            assert isinstance(obj, obj_type) or (
                isinstance(obj, CachingLM) and isinstance(obj.lm, obj_type)
            ), f"Type of retrieved object {obj.__class__} for {obj_name} does not match type {obj_type} specified in DataBuilder {self.__class__}"

            setattr(self, obj_name, obj)

    def __call__(self, instruction_data: List[SdgData]) -> List[SdgData]:
        """Contains the main logic of a data builder. Takes in a list of data objects to be used as seed data and returns a list of data objects that reflect new instances

        Args:
            instruction_data (List[SdgData]): List of data objects to be used as seed data

        Returns:
            List[SdgData]: List of new data objects that can be used for instruction-tuning
        """
        raise NotImplementedError


class TransformationDataBuilder(DataBuilder):
    """Class for transformation data builder"""
