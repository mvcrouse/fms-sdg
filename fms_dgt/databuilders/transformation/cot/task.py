# Standard
from dataclasses import dataclass
from typing import Optional

# Local
from fms_dgt.base.task import SdgData, TransformTask


@dataclass
class CotTransformData(SdgData):

    input: str
    output: str
    thought: Optional[str] = None


class CotTransformTask(TransformTask):

    INPUT_DATA_TYPE = CotTransformData
    OUTPUT_DATA_TYPE = CotTransformData
