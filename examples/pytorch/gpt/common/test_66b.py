import asyncio
import sys
from dacite import from_dict
import timeit
sys.path.append("./")

from together_web3.computer import LanguageModelInferenceRequest
from together_web3.together import TogetherWeb3

async def test():
    together_web3 = TogetherWeb3()
    result = await together_web3.language_model_inference(
        from_dict(
            data_class=LanguageModelInferenceRequest,
            data={
                "model": "opt66b",
                "prompt": "Where is Zurich?",
            }
        ),
    )
    print("result", result)

if __name__=="__main__":
    # measure the end-to-end time
    start = timeit.default_timer()
    asyncio.run(test())
    end = timeit.default_timer()
    print("measure time: {}s".format( (end-start) ))
