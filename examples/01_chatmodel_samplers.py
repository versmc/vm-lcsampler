from dotenv import dotenv_values
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from vm_lcsampler.chatmodel_samplers import (
    ChatModelTextSampler,
    ChatModelStructureSampler,
    ChatModelChunkedTextEnumerator,
)


def execute_text_sampler():
    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),  # type: ignore
        temperature=0.0,  # default to 0.7
    )
    sampler = ChatModelTextSampler(llm=llm)
    generator = sampler.generate(
        category_name="Cat breeds",
        category_description=None,
        few_shot_samples=[
            "Persian",
            "Siamese",
            "Maine Coon",
            "Bengal",
            "Sphynx",
        ],
        num_sample=20,
    )

    for text in generator:
        print(text)


def execute_structure_sampler():
    class Joke(BaseModel):
        setup: str = Field(description="The setup of the joke")  # type: ignore
        punchline: str = Field(description="The punchline to the joke")  # type: ignore

    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),  # type: ignore
        temperature=0.0,  # default to 0.7
    )
    sampler = ChatModelStructureSampler(llm=llm)
    generator = sampler.generate(
        model_name="joke",
        model_description=None,
        schema=Joke,
        few_shot_samples=[
            Joke(
                setup="Why couldn't the bicycle stand up by itself?",
                punchline="It was two tired.",
            ),
            Joke(
                setup="Why did the scarecrow win an award?",
                punchline="Because he was outstanding in his field.",
            ),
            Joke(
                setup="What do you call a fake noodle?",
                punchline="An impasta.",
            ),
        ],
        num_sample=5,
    )

    for joke in generator:
        print(joke.json(ensure_ascii=False, indent=4))


def execute_chunked_text_enumerator():
    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),  # type: ignore
        temperature=0.0,  # default to 0.7
    )
    sampler = ChatModelChunkedTextEnumerator(llm=llm)
    enumerator = sampler.enumerate(
        category_name="Cat breeds",
        category_description=None,
        chunk_size=5,
        num_chunk=4,
        few_shot_chunked_samples=[
            [
                "Persian",
                "Siamese",
                "Maine Coon",
                "Bengal",
                "Sphynx",
            ],
        ],
    )

    for i, text in enumerator:
        print(text)


if __name__ == "__main__":
    if 1:
        execute_text_sampler()
    if 0:
        execute_structure_sampler()
    if 0:
        execute_chunked_text_enumerator()
