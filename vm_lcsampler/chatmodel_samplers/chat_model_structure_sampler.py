from typing import Any, Generator, Type, TypeVar
import json
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel


_ModelField = Any  # temporary solution. I don't know how to import `ModelField` from `langchain_core.pydantic_v1`.


_BM = TypeVar("_BM", bound=BaseModel)


class ChatModelStructureSampler(object):
    """sample structured datas with chat models"""

    _HUMAN_COMMAND = "next"

    @classmethod
    def _create_system_message(
        cls,
        model_name: str,
        model_description: str | None,
        model_fields: dict[str, _ModelField],
    ) -> SystemMessage:
        msg = ""
        msg += (
            f"You are a expert algorithm that provides examples of `{model_name}`."
            f"When a user issues the command `{cls._HUMAN_COMMAND}`, Give one example that is different from any of the existing examples."
            f"\n"
            f"**Answer should include only one example whose unit is `{model_name}`**."
            f"\n"
        )
        msg += "\n"

        if model_description is not None:
            msg += f"## The description of `{model_name}` is shown below.\n"
            msg += f"{model_description}\n"
            msg += "\n"

        msg += f"## The description of fields of `{model_name}` is shown below.\n"
        msg += json.dumps(
            {
                field_name: {
                    "type": field.type_.__name__,
                    "description": field.field_info.description,
                }
                for field_name, field in model_fields.items()
            },
            ensure_ascii=False,
            indent=4,
        )
        msg += "\n"

        return SystemMessage(msg)

    @classmethod
    def _create_human_message(cls) -> HumanMessage:
        return HumanMessage(cls._HUMAN_COMMAND)

    @classmethod
    def _create_sample_ai_message(cls, sample: BaseModel) -> AIMessage:
        return AIMessage(sample.json(indent=4, ensure_ascii=False))

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def sample_n(
        self,
        model_name: str,
        model_description: str | None,
        schema: Type[_BM],
        few_shot_samples: list[_BM] | None,
        num_sample: int,
    ) -> list[_BM]:
        return list(
            self.generate(
                model_name,
                model_description,
                schema,
                few_shot_samples,
                num_sample,
            )
        )

    def generate(
        self,
        model_name: str,
        model_description: str | None,
        schema: Type[_BM],
        few_shot_samples: list[_BM] | None,
        num_sample: int,
    ) -> Generator[_BM, None, None]:
        if few_shot_samples is None:
            few_shot_samples = []

        messages: list[BaseMessage] = []
        messages.append(
            self._create_system_message(
                model_name,
                model_description,
                schema.__fields__,
            )
        )
        for fs_sample in few_shot_samples:
            messages.append(self._create_human_message())
            messages.append(self._create_sample_ai_message(fs_sample))

        llm_with_structured_output = self._llm.with_structured_output(
            schema,
            # include_raw=True
        )
        for _ in range(num_sample):
            messages.append(self._create_human_message())
            new_sample = llm_with_structured_output.invoke(messages)
            if not isinstance(new_sample, schema):
                # This scope is never reached, but is written for static type analysis.
                raise TypeError(
                    f"Unexpected Error. type(new_sample): {type(new_sample)}, schema: {schema}"
                )
            messages.append(AIMessage(new_sample.json(ensure_ascii=False, indent=4)))
            yield new_sample


class ChatModelStructureSamplerJA(ChatModelStructureSampler):
    """Japanese prompt version of `ChatModelStructureSampler`"""

    _HUMAN_COMMAND = "next"

    @classmethod
    def _create_system_message(
        cls,
        model_name: str,
        model_description: str | None,
        model_fields: dict[str, _ModelField],
    ) -> SystemMessage:
        msg = ""
        msg += (
            f"あなたは `{model_name}` の例を例示するエキスパートアルゴリズムです。"
            f"ユーザーがコマンド `{cls._HUMAN_COMMAND}` を発行した場合に, 既存の例のいずれとも異なる例を1つ例示してください。"
            f"\n"
            f"**回答は `{model_name}` を単位として1単位の例のみ示してください**."
            f"\n"
        )
        msg += "\n"

        if model_description is not None:
            msg += f"## `{model_name}` の説明は以下です。\n"
            msg += f"{model_description}\n"
            msg += "\n"

        msg += f"## `{model_name}` の各フィールドの説明は以下です。\n"
        msg += json.dumps(
            {
                field_name: {
                    "type": field.type_.__name__,
                    "description": field.field_info.description,
                }
                for field_name, field in model_fields.items()
            },
            ensure_ascii=False,
            indent=4,
        )
        msg += "\n"

        return SystemMessage(msg)
