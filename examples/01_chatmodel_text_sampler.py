"""
In this script, `ChatModelTextSampler` is used.
However, the `ChatModelChunkedTextEnumerator` is recommended as an alternative.
For more detail, see definition of `ChatModelTextSampler`.
"""

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from vm_lcsampler.chatmodel_samplers import ChatModelTextSampler


def main():
    # Set up a chat model as OpenAI API
    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),  # type: ignore
        temperature=0.0,  # default to 0.7
    )

    # Set up Text sampler
    sampler = ChatModelTextSampler(llm=llm)
    generator = sampler.generate(
        category_name="cat breeds",
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

    # Perform sampling
    for text in generator:
        print(text)


if __name__ == "__main__":
    main()
