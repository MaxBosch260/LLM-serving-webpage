import asyncio
import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch


class _FastAPI:
    def __init__(self, lifespan=None):
        self.lifespan = lifespan

    def get(self, *_args, **_kwargs):
        return lambda endpoint: endpoint

    def post(self, *_args, **_kwargs):
        return lambda endpoint: endpoint


class _BaseModel:
    def __init__(self, **values):
        for field in self.__class__.__annotations__:
            if field in values:
                setattr(self, field, values[field])
            elif hasattr(self.__class__, field):
                setattr(self, field, getattr(self.__class__, field))
            else:
                raise TypeError(f"Missing required field: {field}")


class _FileResponse:
    def __init__(self, path):
        self.path = path


@contextmanager
def _no_grad():
    yield


def _dependency_stubs():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FastAPI

    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.FileResponse = _FileResponse

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = _BaseModel

    torch = types.ModuleType("torch")
    torch.float32 = object()
    torch.no_grad = _no_grad

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type("AutoTokenizer", (), {})
    transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})

    return {
        "fastapi": fastapi,
        "fastapi.responses": fastapi_responses,
        "pydantic": pydantic,
        "torch": torch,
        "transformers": transformers,
    }


with patch.dict(sys.modules, _dependency_stubs()):
    application = importlib.import_module("app")


class _Inputs(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _InputIds:
    shape = (1, 5)


class AppTests(unittest.TestCase):
    def setUp(self):
        application.model_state.clear()

    def tearDown(self):
        application.model_state.clear()

    def test_index_returns_the_application_html_file(self):
        response = application.index()

        self.assertIsInstance(response, _FileResponse)
        self.assertEqual(response.path, Path(application.__file__).with_name("index.html"))

    def test_health_returns_ok_status(self):
        self.assertEqual(application.health(), {"status": "ok"})

    def test_chat_request_defaults_max_new_tokens(self):
        request = application.ChatRequest(message="Hello")

        self.assertEqual(request.message, "Hello")
        self.assertEqual(request.max_new_tokens, 100)

    def test_generate_formats_input_and_decodes_only_generated_tokens(self):
        inputs = _Inputs(input_ids=_InputIds())
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = inputs
        tokenizer.decode.return_value = "Hello!"
        model = Mock()
        model.device = "cpu"
        model.generate.return_value = [[1, 2, 3, 4, 5, 99]]
        application.model_state.update(tokenizer=tokenizer, model=model)

        result = application.generate(
            application.ChatRequest(message="Hi", max_new_tokens=12)
        )

        self.assertEqual(result, {"response": "Hello!"})
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertEqual(inputs.device, "cpu")
        model.generate.assert_called_once_with(
            input_ids=inputs["input_ids"], max_new_tokens=12
        )
        tokenizer.decode.assert_called_once_with([99], skip_special_tokens=True)

    def test_generate_rejects_token_limits_outside_allowed_range(self):
        for token_limit in (0, -1, 2049):
            with self.subTest(token_limit=token_limit):
                with self.assertRaisesRegex(
                    ValueError, "max_new_tokens must be between 1 and 2048"
                ):
                    application.generate(
                        application.ChatRequest(
                            message="Hi", max_new_tokens=token_limit
                        )
                    )

    def test_lifespan_loads_model_and_clears_state_on_shutdown(self):
        tokenizer = Mock()
        model = Mock()
        model.to.return_value = model
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        auto_model = Mock()
        auto_model.from_pretrained.return_value = model

        with (
            patch.object(application, "AutoTokenizer", auto_tokenizer),
            patch.object(application, "AutoModelForCausalLM", auto_model),
        ):
            async def run_lifespan():
                async with application.lifespan(application.app):
                    self.assertIs(application.model_state["tokenizer"], tokenizer)
                    self.assertIs(application.model_state["model"], model)

            asyncio.run(run_lifespan())

        self.assertEqual(application.model_state, {})
        auto_tokenizer.from_pretrained.assert_called_once_with(application.MODEL_PATH)
        auto_model.from_pretrained.assert_called_once_with(
            application.MODEL_PATH, dtype=application.torch.float32
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
