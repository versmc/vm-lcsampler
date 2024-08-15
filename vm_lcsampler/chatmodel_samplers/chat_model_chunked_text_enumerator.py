import ast
from typing import Generator
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage



class ChatModelChunkedTextEnumerator(object):
    """
    This class provide generator to enumerate examples.
    The examples are sampled in chunks.
    In this class, each example is sampled with its index number, which appears to improve accuracy of generated examples.
    """

    @classmethod
    def _create_system_message(
        cls,
        category_name: str,
        category_description: str | None,
        first_index: int,
        last_index: int,
    ) -> SystemMessage:
        return SystemMessage(
            "\n".join(
                [
                    f"You are a expert algorithm that provides examples of `{category_name}`.",
                    *(
                        [
                            f"The description of {category_name} is as follows.",
                            f"{category_description}",
                            "",
                        ]
                        if category_description
                        else []
                    ),
                    f"Please provide examples of `{category_name}`, where the number of examples is given by a user.",
                    "When the user requests as:",
                    str(
                        cls._create_human_message(
                            category_name, first_index, last_index, is_continuous=False
                        ).content
                    ),
                    "The first output format is as follows.",
                    f"{first_index}. <example {first_index}>",
                    f"{first_index+1}. <example {first_index+1}>",
                    "...",
                    f"{last_index}. <example {last_index}>",
                    "",
                    "When user requests more examples, then you shold continue to provide examples",
                ]
            )
        )

    @classmethod
    def _create_human_message(
        cls,
        category_name: str,
        first_index: int,
        last_index: int,
        is_continuous: bool,
    ) -> HumanMessage:
        more = "more" if is_continuous else ""
        return HumanMessage(
            f"Please provide {more} {last_index-first_index+1} examples of {category_name} from {first_index} to {last_index}."
        )

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def _parse_llm_examples(self, text: str, index_list: list[int]) -> dict[int, str]:
        messages = [
            SystemMessage(
                "\n".join(
                    [
                        "You are an expert algorithm that parse a raw text to a json with format:",
                        "{",
                        '    "<index>": "<content>",',
                        "    ...",
                        "}.",
                        "Please return only the json result.",
                    ]
                )
            ),
            HumanMessage(text),
        ]

        ai_message = self._llm.invoke(messages)
        ai_json = str(ai_message.content)
        while ai_json[: len("\n")] == "\n":
            ai_json = ai_json[len("\n") :]
        while ai_json[-len("\n") :] == "\n":
            ai_json = ai_json[: -len("\n")]
        while ai_json[: len("```json")] == "```json":
            ai_json = ai_json[len("```json") :]
        while ai_json[: len("```")] == "```":
            ai_json = ai_json[len("```") :]
        while ai_json[-len("```") :] == "```":
            ai_json = ai_json[: -len("```")]
        obj = ast.literal_eval(ai_json)
        if not isinstance(obj, dict):
            raise TypeError(f"Parse error. `type(obj): {type(obj)}` is not dict.")

        obj = {int(k): str(v) for k, v in obj.items()}
        if set(obj.keys()) != set(index_list):
            raise ValueError(
                "\n".join(
                    [
                        "set(obj.keys()) != set(index_list)",
                        f"set(obj.keys()): {set(obj.keys())}",
                        f"set(index_list): {set(index_list)}",
                    ]
                )
            )
        return obj

    def generate_chunk(
        self,
        category_name: str,
        category_description: str | None,
        chunk_size: int,
        num_chunk: int,
        few_shot_chunked_samples: list[list[str]] | None,
    ) -> Generator[dict[int, str], None, None]:
        if few_shot_chunked_samples is None:
            few_shot_chunked_samples = []

        for chunked_sample in few_shot_chunked_samples:
            if len(chunked_sample) != chunk_size:
                raise ValueError(
                    f"len(chunked_sample) == chunk_size must be satisfied. len(chunked_sample): {len(chunked_sample)}, num_chunk: {chunk_size}"
                )

        messages: list[BaseMessage] = [
            self._create_system_message(
                category_name,
                category_description,
                first_index=1,
                last_index=chunk_size,
            ),
        ]
        for i_chunk in range(len(few_shot_chunked_samples) + num_chunk):
            first = i_chunk * chunk_size + 1
            last = (i_chunk + 1) * chunk_size
            messages.append(
                self._create_human_message(
                    category_name=category_name,
                    first_index=first,
                    last_index=last,
                    is_continuous=(i_chunk != 0),
                )
            )
            ai_text = ""
            if i_chunk < len(few_shot_chunked_samples):
                sample_ai_message: BaseMessage = AIMessage(
                    "\n".join(
                        [
                            f"{first+i}. {sample}"
                            for i, sample in enumerate(
                                few_shot_chunked_samples[i_chunk]
                            )
                        ]
                    )
                )
                messages.append(sample_ai_message)
                ai_text = str(sample_ai_message.content)
            else:
                ai_message: BaseMessage = self._llm.invoke(messages)
                messages.append(ai_message)
                ai_text = str(ai_message.content)
                ai_dict = self._parse_llm_examples(
                    text=ai_text,
                    index_list=[i for i in range(first, last + 1, 1)],
                )
                yield ai_dict

    def generate(
        self,
        category_name: str,
        category_description: str | None,
        chunk_size: int,
        num_chunk: int,
        few_shot_chunked_samples: list[list[str]] | None,
    ) -> Generator[str, None, None]:
        chunk_gen = self.generate_chunk(
            category_name,
            category_description,
            chunk_size,
            num_chunk,
            few_shot_chunked_samples,
        )
        for chunk in chunk_gen:
            for text in chunk.values():
                yield text

    def enumerate(
        self,
        category_name: str,
        category_description: str | None,
        chunk_size: int,
        num_chunk: int,
        few_shot_chunked_samples: list[list[str]] | None,
    ) -> Generator[tuple[int, str], None, None]:
        generator = self.generate(
            category_name,
            category_description,
            chunk_size,
            num_chunk,
            few_shot_chunked_samples,
        )
        for i, e in enumerate(generator):
            yield (i, e)

    def sample_n(
        self,
        category_name: str,
        category_description: str | None,
        chunk_size: int,
        num_chunk: int,
        few_shot_chunked_samples: list[list[str]] | None,
    ) -> list[str]:
        return [
            text
            for text in self.generate(
                category_name,
                category_description,
                chunk_size,
                num_chunk,
                few_shot_chunked_samples,
            )
        ]
