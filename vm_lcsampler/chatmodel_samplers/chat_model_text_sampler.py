from typing import Generator
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class ChatModelTextSampler(object):
    """
    sampling texts with chat models.

    ** Note **
    In this class, examples are sampled by repeating a single generation.
    While this algorithm is simple,  it is empirically known to introduce sampling bias.
    As an alternative, `ChatModelChunkedTextEnumerator` offers the same functionality,
    which is emprically more robust against the sampling bias.
    """

    _HUMAN_COMMAND = "next"

    @classmethod
    def _create_system_message(
        cls, category_name: str, category_description: str | None
    ) -> SystemMessage:
        msg = ""
        msg += (
            f"You are a expert algorithm that provides examples of `{category_name}`."
            f"When a user issues the command `{cls._HUMAN_COMMAND}`, Give one example that is different from any of the existing examples."
            f"\n"
            f"**Answer should include only one example whose unit is `{category_name}`**."
            f"\n"
        )
        if category_description is not None:
            msg += f"The description of `{category_name}` is shown below.\n"
            msg += f"{category_description}\n"
        return SystemMessage(msg)

    @classmethod
    def _create_human_message(cls) -> HumanMessage:
        return HumanMessage(cls._HUMAN_COMMAND)

    @classmethod
    def _create_sample_ai_message(cls, sample: str) -> AIMessage:
        return AIMessage(sample)

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def sample_n(
        self,
        category_name: str,
        category_description: str | None,
        few_shot_samples: list[str] | None,
        num_sample: int,
    ) -> list[str]:
        return list(
            self.generate(
                category_name, category_description, few_shot_samples, num_sample
            )
        )

    def generate(
        self,
        category_name: str,
        category_description: str | None,
        few_shot_samples: list[str] | None,
        num_sample: int,
    ) -> Generator[str, None, None]:
        if few_shot_samples is None:
            few_shot_samples = []
        messages: list[BaseMessage] = []
        messages.append(
            self._create_system_message(
                category_name=category_name,
                category_description=category_description,
            )
        )
        for fs_sample in few_shot_samples:
            messages.append(self._create_human_message())
            messages.append(self._create_sample_ai_message(fs_sample))
        for _ in range(num_sample):
            messages.append(self._create_human_message())
            new_ai_message = self._llm.invoke(messages)
            messages.append(new_ai_message)
            yield str(new_ai_message.content)


class ChatModelTextSamplerJA(ChatModelTextSampler):
    """Japanese version of `ChatModelTextSampler`"""

    @classmethod
    def _create_system_message(
        cls, category_name: str, category_description: str | None
    ) -> SystemMessage:
        msg = ""
        msg += (
            f"あなたは `{category_name}` の例を挙げるエキスパートアルゴリズムです。"
            f"ユーザーが `{cls._HUMAN_COMMAND}` というコマンドを発行した場合に、既存の例と異なる例を1つ挙げてください。"
            "**回答は例を1つのみ含めてください**。\n"
        )
        if category_description is not None:
            msg += f"`{category_name}` についての説明は以下になります。\n"
            msg += f"{category_description}\n"
        return SystemMessage(msg)
