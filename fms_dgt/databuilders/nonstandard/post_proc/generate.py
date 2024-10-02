# Standard
from typing import Iterable, List

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgData, SdgTask


@register_data_builder("post_proc_only")
class PostProcessingOnlyDataBuilder(TransformationDataBuilder):
    """Class for databuilder that only executes postprocessing"""

    def call_with_task_list(self, tasks: List[SdgTask]) -> Iterable[SdgData]:
        data_pool = [e for task in tasks for e in task.get_batch_examples()]
        while data_pool:
            for inp in data_pool:
                yield inp
            data_pool = [e for task in tasks for e in task.get_batch_examples()]
