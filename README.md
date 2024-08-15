# vm-lcsampler: LLM Sampler with LangChain chat models.

## Overview
This repository provides tools for sampling text and structured data using Large Language Models (LLMs) by prompt engineering on [LangChain](https://python.langchain.com/v0.2/docs/introduction/).


## Install
This repository is managed using [poetry](https://python-poetry.org/). You can install with the following command.
```sh
pip install -e .
```

## Examples
In the following examples, OpenAI API key must be set in `.env` file according to `./.env.org`.
### Raw text sampling
The full code is in [examples/02_chatmodel_structure_sampler.py](examples/02_chatmodel_structure_sampler.py).  

```python
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from vm_lcsampler.chatmodel_samplers import ChatModelChunkedTextEnumerator

# Set up LLM
llm = ChatOpenAI(
    api_key=str(dotenv_values()["OPENAI_API_KEY"]),
    temperature=0.0,  # default to 0.7
)

# Set up sampler
sampler = ChatModelChunkedTextEnumerator(llm=llm)
enumerator = sampler.enumerate(
    category_name="cat breeds",
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
```


### Structured object sampling
See [examples/02_chatmodel_structure_sampler.py](examples/02_chatmodel_structure_sampler.py)