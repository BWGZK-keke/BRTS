# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl.utils.model import load_valuehead_model


@patch("transformers.AutoModelForTokenClassification.from_pretrained")
def test_load_valuehead_model_respects_model_config_attn_impl(mock_from_pretrained):
    """Ensure we don't hardcode flash_attention_2 when config requests another impl."""
    expected_model = object()
    mock_from_pretrained.return_value = expected_model

    model_config = SimpleNamespace(_attn_implementation="eager")
    model = load_valuehead_model(
        local_path="/tmp/fake_model",
        torch_dtype=torch.float16,
        model_config=model_config,
        trust_remote_code=False,
    )

    assert model is expected_model
    kwargs = mock_from_pretrained.call_args.kwargs
    assert kwargs["attn_implementation"] == "eager"


@patch("verl.utils.model._resolve_attn_implementation_for_hf_loading", return_value="sdpa")
@patch("transformers.AutoModelForTokenClassification.from_pretrained")
def test_load_valuehead_model_uses_resolved_attn_impl(mock_from_pretrained, _mock_resolver):
    """Ensure resolved fallback impl (e.g. sdpa) is used for model loading."""
    expected_model = object()
    mock_from_pretrained.return_value = expected_model

    model_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    model = load_valuehead_model(
        local_path="/tmp/fake_model",
        torch_dtype=torch.bfloat16,
        model_config=model_config,
        trust_remote_code=False,
    )

    assert model is expected_model
    kwargs = mock_from_pretrained.call_args.kwargs
    assert kwargs["attn_implementation"] == "sdpa"
