from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self,retriever,api_key:str,model_name:str):
        self.llm = ChatGroq(api_key=api_key,model=model_name,temperature=0)
        self.prompt = get_anime_prompt()

        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_recommendation(self, query:str):
        result = self.qa_chain.invoke(query)
        return result