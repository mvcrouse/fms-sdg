# Standard
from typing import Any, Iterable, List

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.databuilders.transformation.cot.task import (
    CotTransformData,
    CotTransformTask,
)

_PROMPT = """You are an intelligent tutoring assistant explains how to provide an answer. Given a question and its answer, explain how to solve the question step-by-step to achieve the answer. When you are explaining the answer to the student, please preface your explanation with "Let's think step-by-step."

Here are some examples:

Question: { { question } }
Answer: { { answer } }
Explanation: Let's think step-by-step. 
""".strip()


# NOTE: importantly, transformation data builders are STRONGLY ENCOURAGED to inherit from TransformationDataBuilder, rather than from BaseDataBuilder. This is because the default behavior of TransformationDataBuilder is much more suited to transformation tasks
@register_data_builder("cot_transform")
class Gsm8kCotTransformDataBuilder(TransformationDataBuilder):
    """Class for GSM8K chain-of-thought task"""

    TASK_TYPE: CotTransformTask = CotTransformTask

    # NOTE: this is the same llm1 as in our config yaml
    llm1: LMGenerator

    # NOTE: this can be removed, but we've kept it for those who will copy-paste this as a template
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(
        self,
        qa_data: List[CotTransformData],
    ) -> Iterable[CotTransformData]:

        llm_inputs = []
        for qa_pair in tqdm(qa_data, desc="CoT Transformation"):
            # NOTE: since we have obtained this from huggingface, the actual answer is marked by "... #### <number>", so we'll extract that here

            new_inp = _PROMPT.replace("{ { question } }", qa_pair.input).replace(
                "{ { answer } }", qa_pair.output
            )
            llm_inputs.append(
                {"prompt": new_inp, "stop_sequences": ["Question:"], "data": qa_pair}
            )

        # NOTE: unlike in the other tutorials, we have provided 'arg_fields' / 'kwarg_fields' / 'result_field' in the data builder's config, thus we do not need to specify them here
        llm_outputs = self.llm1.generate(llm_inputs)

        for output in llm_outputs:
            orig_qa: CotTransformData = output["data"]
            # NOTE: we don't do any validation of the generated 'thought', however, in general that would be a good idea
            thought = output["result"].strip()
            # NOTE: here we yield from the data builder so that the data is saved immediately
            yield CotTransformData(
                **{
                    "task_name": orig_qa.task_name,
                    "input": orig_qa.input,
                    "output": orig_qa.output,
                    "thought": thought,
                }
            )
