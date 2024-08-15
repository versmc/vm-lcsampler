from dotenv import dotenv_values
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from vm_lcsampler.chatmodel_samplers import ChatModelStructureSampler


def main():
    # Define Structure to sample
    class Joke(BaseModel):
        setup: str = Field(description="The setup of the joke")  # type: ignore
        punchline: str = Field(description="The punchline to the joke")  # type: ignore

    # Set up LLM
    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),  # type: ignore
        temperature=0.0,  # default to 0.7
    )

    # Set up sampler
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

    # Perform sampling
    for joke in generator:
        print(joke.json(ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
