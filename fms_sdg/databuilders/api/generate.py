# Standard
from typing import Any, List, Optional, Tuple
import copy
import random
import time

# Local
from fms_sdg.base.databuilder import DataBuilder
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_data_builder
from fms_sdg.base.task import group_data_by_task
from fms_sdg.databuilders.api.task import ApiSdgData, ApiSdgTask
from fms_sdg.generators.llm import LMGenerator
from fms_sdg.utils import sdg_logger
from fms_sdg.validators.api import APIGenSpecValidator, ApiGenSpecYesNoValidation
from fms_sdg.validators.rouge import RougeValidator
import fms_sdg.databuilders.api.utils as api_utils


class ApiDataBuilder(DataBuilder):
    """Class for API Sequence task"""

    TASK_TYPE: ApiSdgTask = ApiSdgTask

    def __init__(
        self,
        *args: Any,
        num_prompt_instructions: Optional[int] = 3,
        num_base_examples: Optional[int] = 10,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self._num_prompt_instructions = num_prompt_instructions
        self._num_base_examples = num_base_examples
        assert (
            self._num_prompt_instructions >= 1
        ), "Number of prompt examples must be at least 1"

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator
    val1: APIGenSpecValidator
    val2: RougeValidator

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[ApiSdgData],
    ) -> List[ApiSdgData]:

        # first generate new data
        instruction_data = instruction_data + []
        random.shuffle(instruction_data)
        gen_inputs: List[Instance] = []
        for task_data in group_data_by_task(instruction_data):
            for _ in range(self._num_base_examples):
                prompt, new_instr = self._construct_new_data(task_data)
                args = [prompt]
                kwargs = {"stop_sequences": [f"API:"]}
                gen_inputs.append(Instance(args, kwargs, data=new_instr))

        request_start = time.time()
        self.llm1.generate_batch(gen_inputs)
        request_duration = time.time() - request_start

        # now begin filtering generated data
        post_process_start = time.time()

        outputs, wf_discarded = self._wf_filter_data(gen_inputs)

        outputs, rouge_discarded = self._rouge_filter_data(outputs)

        # return
        post_process_duration = time.time() - post_process_start
        sdg_logger.debug(
            "Request %s took %.2fs, post-processing took %.2fs, discarded %s instances",
            request_idx,
            request_duration,
            post_process_duration,
            wf_discarded + rouge_discarded,
        )

        return outputs

    def _wf_filter_data(self, data_to_filter: List[Instance]):
        # Well-formedness filtering
        val1_inputs: List[Instance] = []
        discarded = 0
        for gen_inp in data_to_filter:
            new_instr: ApiSdgData = gen_inp.data
            components = gen_inp.result.split("A:")
            if len(components) == 2:
                question, answer = [x.strip() for x in components]
                new_instr.input = question
                new_instr.output = answer
                new_apis = {
                    pos_func: new_instr.api_specifications[new_instr.seed_api_group][
                        pos_func
                    ]
                    for pos_func in new_instr.positive_functions
                }

                # grab schema from input
                args = [new_apis, question, answer]
                kwargs = {
                    "check_arg_question_overlap": new_instr.check_arg_question_overlap,
                    "intent_only": new_instr.intent_only,
                    "require_nested": new_instr.require_nested,
                    "min_ct": (
                        new_instr.func_count_bounds[0]
                        if new_instr.single_function
                        else len(new_instr.positive_functions)
                    ),
                    "max_ct": (
                        new_instr.func_count_bounds[1]
                        if new_instr.single_function
                        else len(new_instr.positive_functions)
                    ),
                }
                val1_inputs.append(Instance(args, kwargs, data=new_instr))
            else:
                discarded += 1

        self.val1.validate_batch(val1_inputs)

        # filter invalid data
        outputs: List[ApiSdgData] = [
            val1_input.data for val1_input in val1_inputs if val1_input.result
        ]
        discarded += len(val1_inputs) - len(outputs)

        return outputs, discarded

    def _rouge_filter_data(self, data_to_filter: List[ApiSdgData]):
        # Rouge filtering
        all_instruction_tokens = self.val2.tokenize(
            [instr.input for instr in data_to_filter]
        )

        val2_inputs: List[Instance] = []
        for new_data in data_to_filter:
            # computing similarity with the pre-tokenized instructions
            new_instruction_tokens = self.val2.tokenize(new_data.input)
            args = [new_instruction_tokens, all_instruction_tokens]
            val2_inputs.append(Instance(args, data=new_data))
        self.val2.validate_batch(val2_inputs)

        # filter rouge failed data
        outputs = [val2_input.data for val2_input in val2_inputs if val2_input.result]
        discarded = len(val2_inputs) - len(outputs)

        return outputs, discarded

    def _construct_new_data(self, task_data: List[ApiSdgData]):
        # gather ICL examples
        base_instr = task_data[0]
        groups = list(base_instr.api_specifications.keys())
        random.shuffle(groups)
        grouped_data: List[ApiSdgData] = []
        for group in groups:
            avail_data = [td for td in task_data if td.seed_api_group == group]
            if avail_data:
                grouped_data.append(random.choice(avail_data))

        grouped_data = grouped_data[: random.randint(1, self._num_prompt_instructions)]

        prompt_strings = [grouped_data[0].instruction]
        for instr in grouped_data:
            # TODO: cache string transform
            instr_api_specification = api_utils.api_spec_to_str(
                instr.api_specifications[instr.seed_api_group],
                instr.positive_functions,
                instr.task_name,
            )
            prompt_strings.append(
                f"API:\n{instr_api_specification}\nQ: {instr.input}\nA: {instr.output}"
            )

        # now build new example, we'll copy instr and clear its fields to be safe
        new_instr = instr.make_clear_copy()

        # just use last instruction to select new seed_api_group and positive_functions
        key_lst, key_weights = zip(
            *[
                (k, len(v))
                for k, v in new_instr.api_specifications.items()
                # if k not in [gd.seed_api_group for gd in grouped_data]
            ]
        )
        new_group = random.choices(key_lst, weights=key_weights, k=1)[0]
        new_pos_ct = random.randint(*new_instr.func_count_bounds)
        new_pos_apis = random.sample(
            list(new_instr.api_specifications[new_group]),
            k=new_pos_ct,
        )

        # when we have single function, we'll just take the first as the target
        if new_instr.single_function:
            new_pos_apis = [new_pos_apis[0]]

        new_api_specification = api_utils.api_spec_to_str(
            new_instr.api_specifications[new_group],
            new_pos_apis,
            new_instr.task_name,
        )

        new_instr.positive_functions = new_pos_apis
        new_instr.seed_api_group = new_group

        prompt_strings.append(f"API:\n{new_api_specification}\nQ:")
        prompt = "\n\n".join(prompt_strings)

        return prompt, new_instr


@register_data_builder("api_yes_no_detection")
class ApiYesNoDataBuilder(ApiDataBuilder):
    """Class for API Sequence task"""

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator
    val1: ApiGenSpecYesNoValidation


@register_data_builder("api_function_checking")
class ApiDetectionDataBuilder(ApiDataBuilder):
    """Class for API Sequence task"""

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator
    val1: APIGenSpecValidator
