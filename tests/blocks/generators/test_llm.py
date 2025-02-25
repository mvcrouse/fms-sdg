# Standard
from typing import Dict, List
import copy
import os
import time

# Third Party
import pytest

# Local
from fms_dgt.base.registry import get_block
from fms_dgt.blocks.generators.llm import CachingLM, LMBlockData, LMGenerator

#

GREEDY_CFG = {
    "decoding_method": "greedy",
    "temperature": 1.0,
    "max_new_tokens": 5,
    "min_new_tokens": 1,
    "n": 3,
}
GREEDY_GENAI_CFG = {
    "type": "genai",
    "model_id_or_path": "ibm/granite-8b-code-instruct",
    **GREEDY_CFG,
}
GREEDY_VLLM_CFG = {
    "type": "vllm",
    "model_id_or_path": "ibm-granite/granite-8b-code-instruct",
    "model_id_or_path": "mistralai/Mistral-7B-Instruct-v0.1",
    "tensor_parallel_size": 1,
    **GREEDY_CFG,
    "n": 1,
}
GREEDY_VLLM_SERVER_CFG = {
    "type": "vllm-server",
    "model_id_or_path": "ibm-granite/granite-8b-code-instruct",
    "tensor_parallel_size": 1,
    **GREEDY_CFG,
}
GREEDY_OPENAI_CFG = {
    "type": "openai-chat",
    "model_id_or_path": "gpt-3.5-turbo",
    **GREEDY_CFG,
}
GREEDY_WATSONX_CFG = {
    "type": "watsonx",
    "model_id_or_path": "ibm/granite-3-8b-instruct",
    **GREEDY_CFG,
}
PROMPTS = [f"Question: x = {i} + 1\nAnswer: x =" for i in range(25)]


@pytest.mark.parametrize(
    "model_cfg",
    [
        # GREEDY_VLLM_SERVER_CFG,
        GREEDY_VLLM_CFG,
        GREEDY_GENAI_CFG,
        GREEDY_OPENAI_CFG,
    ],
)
def test_generate_batch(model_cfg):
    model_cfg = dict(model_cfg)
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs)

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt"] == inputs_copy[i]["prompt"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"], str) or (
            isinstance(inp["result"], list) and len(inp["result"]) == model_cfg["n"]
        )


@pytest.mark.parametrize("model_cfg", [GREEDY_GENAI_CFG])  # , GREEDY_VLLM_CFG])
def test_loglikelihood_batch(model_cfg):
    model_cfg = dict(model_cfg)
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt, "continuation": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs)

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt"] == inputs_copy[i]["prompt"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"], float)


# def test_loglikelihood_batch_alignment():
#     vllm_config, genai_config = dict(GREEDY_VLLM_CFG), dict(GREEDY_GENAI_CFG)
#     vllm_config["model_id_or_path"] = "ibm-granite/granite-8b-code-instruct"
#     genai_config["model_id_or_path"] = "ibm/granite-8b-code-instruct"

#     vllm: LMGeneratorBlock = get_block(vllm_config["type"],
#         name=f"test_{vllm_config['type']}", config=vllm_config
#     )
#     genai: LMGeneratorBlock = get_block(genai_config["type"],
#         name=f"test_{genai_config['type']}", config=genai_config
#     )

#     inputs: List[Instance] = []
#     for prompt in PROMPTS[:1]:
#         args = [prompt, prompt]
#         inputs.append(Instance(args))

#     inputs_vllm = copy.deepcopy(inputs)
#     inputs_genai = copy.deepcopy(inputs)

#     vllm.loglikelihood_batch(inputs_vllm)
#     genai.loglikelihood_batch(inputs_genai)

#     for i, inp in enumerate(inputs):
#         assert (
#             inp.args == inputs_vllm[i].args == inputs_genai[i].args
#         ), f"Input list has been rearranged at index {i}"


def test_lm_caching():
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    model_cfg = dict(GREEDY_WATSONX_CFG)
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    non_cache_inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        non_cache_inputs.append(inp)

    pre_cache_inputs = copy.deepcopy(non_cache_inputs)
    post_cache_inputs = copy.deepcopy(non_cache_inputs)

    non_cache_time = time.time()
    lm(non_cache_inputs)
    non_cache_time = time.time() - non_cache_time

    cache_lm = CachingLM(
        lm,
        force_cache=False,
        cache_db=cache_path,
    )

    pre_cache_time = time.time()
    cache_lm(pre_cache_inputs)
    pre_cache_time = time.time() - pre_cache_time

    post_cache_time = time.time()
    cache_lm(post_cache_inputs)
    post_cache_time = time.time() - post_cache_time

    os.remove(cache_path)

    assert (
        post_cache_time < pre_cache_time and post_cache_time < non_cache_time
    ), f"Caching led to increased execution time {(post_cache_time, pre_cache_time, non_cache_time)}"

    for i, (non, pre, post) in enumerate(
        zip(non_cache_inputs, pre_cache_inputs, post_cache_inputs)
    ):
        assert (
            non["prompt"] == pre["prompt"] == post["prompt"]
        ), f"Input list has been rearranged at index {i}: {(non['prompt'], pre['prompt'], post['prompt'])}"
        assert (
            non["result"] == pre["result"] == post["result"]
        ), f"Different results detected at index {i}: {(non['output'], pre['output'], post['output'])}"


def test_vllm_remote_batch():
    """
    start server with

    python -m vllm.entrypoints.openai.api_server --model ibm-granite/granite-8b-code-instruct

    """
    model_cfg = dict(GREEDY_VLLM_CFG)
    model_cfg["type"] = "vllm-remote"
    model_cfg["base_url"] = "http://0.0.0.0:8000/v1"
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs)

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt"] == inputs_copy[i]["prompt"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"], str)

    # now we test loglikelihood
    # inputs: List[Dict] = []
    # for prompt in PROMPTS:
    #     inp = {"prompt1": prompt, "prompt2": prompt}
    #     inputs.append(inp)

    # inputs_copy = copy.deepcopy(inputs)

    # lm(
    #     inputs,
    #     arg_fields=["prompt1", "prompt2"],
    #     result_field="result",
    #     method="loglikelihood",
    # )

    # for i, inp in enumerate(inputs):
    #     assert (
    #         inp["prompt1"] == inputs_copy[i]["prompt1"]
    #     ), f"Input list has been rearranged at index {i}"
    #     assert isinstance(inp["result"], float)


def test_vllm_tensor_parallel():
    """

    replace "model_id_or_path" with suitably large model and ensure you have 2 GPUs of sufficient size, e.g. 2 of the a100_80gb

    """
    model_cfg = dict(GREEDY_VLLM_CFG)
    model_cfg["type"] = "vllm"
    model_cfg["tensor_parallel_size"] = 2
    model_cfg["model_id_or_path"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm(inputs)

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt"] == inputs_copy[i]["prompt"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["result"], str)


@pytest.mark.parametrize("model_cfg", [GREEDY_GENAI_CFG, GREEDY_OPENAI_CFG])
def test_auto_chat_template(model_cfg):
    model_type = model_cfg.get("type")
    model_cfg["auto_chat_template"] = True
    model_cfg["model_id_or_path"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # check it passes through for a simple string
    prompt = {"prompt": "Hello world"}
    output = lm._adjust_prompts(
        [LMBlockData(prompt=prompt, SRC_DATA=None)], lm.GENERATE
    )[0]
    if "openai" in model_type:
        assert output == prompt["prompt"]
    else:
        assert output != prompt["prompt"]

    # check it passes through a list of dictionaries
    prompt = {
        "prompt": [
            {"role": "user", "content": "Hello World"},
            {"role": "assistant", "content": "Yes, it is me, World"},
        ]
    }
    output = lm._adjust_prompts(
        [LMBlockData(prompt=prompt, SRC_DATA=None)], lm.GENERATE
    )[0]
    if "openai" in model_type:
        assert output == prompt["prompt"]
    else:
        assert output != prompt["prompt"]

    # check it does nothing for loglikelihood
    prompt = {"prompt": "Hello world"}
    output = lm._adjust_prompts(
        [LMBlockData(prompt=prompt, SRC_DATA=None)], lm.LOGLIKELIHOOD
    )[0]
    assert output == prompt["prompt"]
