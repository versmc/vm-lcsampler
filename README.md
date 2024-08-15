# vm-lcsampler

## Overview
Sampling with LLM through [LangChain](https://python.langchain.com/v0.2/docs/introduction/) chat models.


## Examples

### text sampling
```python
from dotenv import dotenv_values
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from vm_lcsampler.chatmodel_samplers import (
    ChatModelTextSampler,
    ChatModelStructureSampler,
    ChatModelChunkedTextEnumerator,
)

sampler = ChatModelChunkedTextEnumerator(
    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),
        temperature=0.7,
    )
)
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
```

### structured object sampling
```python
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


sampler = ChatModelStructureSampler(
    llm = ChatOpenAI(
        api_key=str(dotenv_values()["OPENAI_API_KEY"]),
        temperature=0.7,
    )
)
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

```


### more

See [./examples/01_chatmodel_samplers.py](./examples/01_chatmodel_samplers.py) for full examples.