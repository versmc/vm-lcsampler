from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from vm_lcsampler.chatmodel_samplers import ChatModelChunkedTextEnumerator



def main():

    # Set up LLM
    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),  # type: ignore
        temperature=0.0,  # default to 0.7
    )

    # Set up sampler
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

    # Perform sampling
    for i, text in enumerator:
        print(text)


if __name__ == "__main__":
    main()
