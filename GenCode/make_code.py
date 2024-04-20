from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse

class TestLLM:
    
    def __init__(self):

        self._llm = Ollama(
            base_url='http://localhost:11434',
            model='mistral'
        )

    def create_chain(self, template: str, input_variables: list, output_key: str) -> LLMChain:
        '''
            Constructs a LLM Chain given a prompt sequence.
        '''

        template = PromptTemplate(
            template = template,
            input_variables = input_variables
        )

        return LLMChain(
            llm=self._llm,
            prompt=template,
            output_key=output_key
        )


    def get_llm_result(self, query: str) -> str:
        '''
            Gets the output from the llm given a query string.
        '''
        return self._llm.invoke(query)

if __name__ == "__main__":
    tLLM = TestLLM()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="return a list of numbers")
    parser.add_argument("--language", default="python")
    args = parser.parse_args()

    '''
        1. Sample langchain querying.
    '''

    # query = str.strip(input())
    # print(tLLM.get_llm_result(query))

    '''
        2. Creating chains.
    '''

    # create a code-generating chain
    code_chain = tLLM.create_chain(
        template = "Answer using the code only: create a short {language} function that will {task}.",
        input_variables = ["language", "task"],
        output_key = "code"
    )

    # create a test-generating chain given the inputs of code_chain
    test_chain = tLLM.create_chain(
        template = "Write a test for the following {language} code: {code}.",
        input_variables = ["language", "code"],
        output_key = "test"
    )

    '''
        2.1 Manually connecting two chains together (i.e. pipe the input from code_chain into test_chain)
    '''

    # returns a dictionary that contains the results 
    # code_result = code_chain({
    #     "language": args.language,
    #     "task": args.task
    # })

    # manually feed the output of code_chain into test_chain 
    # test_result = test_chain({
    #     "language": args.language,
    #     "code": code_result['code']
    # })

    # printing the results of both test
    # print(code_result['code'])
    # print(test_result['test'])


    '''
        2.1 Manually connecting two chains together (i.e. pipe the input from code_chain into test_chain)
    '''

    seqchain = SequentialChain(
        chains=[code_chain, test_chain],
        input_variables = ["language", "task"],
        output_variables = ["code", "test"]
    )

    code_result = seqchain({
        "language": args.language,
        "task": args.task
    })


    print("---------------------Code generated---------------------")
    print(code_result["code"])
    print("---------------------Test generated---------------------")
    print(code_result["test"])
