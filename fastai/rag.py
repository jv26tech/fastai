from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

from operator import itemgetter
from fastai.settings import Settings
from fastai.qdrant import vector_store

settings = Settings()

model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=settings.GOOGLE_API_KEY,
    temperatur=0.3,
    convert_system_message_to_human=True,
)

prompt_template = """
Answer the question based on the context, in a concise manner and using bullet points where applicable.

Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

retriever = vector_store.as_retriever()


def create_chain():
    chain = {
        "context": retriever.with_config(top_k=3),
        "question": RunnablePassthrough(),
    } | RunnableParallel({"response": prompt | model, "context": itemgetter("context")})
    return chain


def get_answer_and_docs(question: str):
    chain = create_chain()
    response = chain.invoke(question)
    answer = response["response"].content
    context = response["context"]

    return {"answer": answer, "context": context}


# response = get_answer_and_docs(
#     'What are the authos mentioned in the page?'
# )

# print(response)