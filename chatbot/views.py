from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
import os
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import langchain
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import Tool

import warnings
warnings.filterwarnings("ignore")
import re
import string
from langchain.agents import initialize_agent
from langchain.agents import AgentType
# from google.colab import drive
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="llama3-70b-8192", temperature=0)
model_name="Alibaba-NLP/gte-large-en-v1.5"
# model_name="WhereIsAI/UAE-Large-V1"
hf=HuggingFaceEmbeddings(model_name=model_name,
                         model_kwargs={'trust_remote_code':True})

# Dummy response generator

rag_prompt=PromptTemplate(
    template= """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    If the {question} is not a question, respond as a human would.
    You are an assistant for question-answering tasks employed by I4di to answer questions regarding their website information.
    Utilize the following retrieved context to construct your answers.
    Your writing should be accessible, targeting a 10th-grade reading level.
    It is essential that your communication is impactful and elaborate.
    Ensure that your responses are rich in content and details.
    Always use the document context to create your responses.
    If you don't know the answer, just say that you don't know.
    Do not provide an answer based on your knowledge. If answers cannot be answered from the context provided, return that answers cannot be answered from the context.
    Begin your responses as if you are directly addressing the question rather than referencing the context. For example, instead of starting with "Based on information provided, the CEO is ...", phrase it as "The CEO is ...
     <|eot_id|><|start_header_id|>user<|end_user_id|>
    Question:{question}
    Context:{context}
    Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"])
# format the document and src
def format_con(docs):
  return " ".join(doc.page_content for doc in docs)
def format_src(docs):
  return "\n\n".join(doc.metadata["source"].split("/")[-1] for doc in docs)
rag_chain=rag_prompt | llm | StrOutputParser()

cojoin_question = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Reformulate the latest user question so that it can be understood independently of the chat history. 
Do not provide an answer—focus solely on rephrasing the question if needed without any explanation or preamble.


Rules and Guidelines
Vague Questions:


For vague questions (e.g., “What happened?” or “Answer this question”):
Refer to the most recent unanswered or related question from the chat history.
Reformulate the question by incorporating enough context for clarity.
Contextual References:


If the user question hints at or builds upon a previous question:
Include the necessary context from the chat history in the reformulated question.

Instruction-Based Questions: If the question is an instruction (e.g., “Write this as that” or “Make this like this”): Return it unchanged. 


Standalone Questions: If the question does not reference the chat history or is unrelated to it:Return the question unchanged.


Table Requests: If the question implies a need for tabular output, reformulate explicitly to prompt a table: Example: “Can you list the metrics for this project?” Example: “provide a structured response for this question” → “Could you create a table listing the metrics for this project?”
Decipher: If the question talks about Decipher, do not put it in quotes as Decipher is a product of I4DI

Tone and Clarity: Maintain a conversational tone in all reformulated questions.
Ensure the question provides enough context for standalone understanding without being verbose.


Additional Considerations
Handling Ambiguity: If the user question is highly ambiguous and cannot be resolved from the chat history, return it unchanged but note internally that the ambiguity remains.
Avoid Over-Reformulation: Do not add unnecessary detail or complexity to the reformulated question.
Preserve Original Intent: Ensure that the reformulated question aligns fully with the user’s intent and does not alter its meaning.
Examples


Chat History:
Q1: “What are the goals of the project?”
Q2: “How does this align with the broader objectives?”
Latest Question:


“What does this mean?”
Reformulated Question:
“What does the alignment between the project goals and broader objectives mean?”


Chat History:
Q1: “What indicators are being tracked for the program?”
Latest Question:
“Make this a table.”
Reformulated Question:
“Could you create a table listing the indicators being tracked for the program?”
Chat History:


(No prior chat history)
Latest Question:


“How do I conduct an evaluation?”
Reformulated Question:
(Unchanged) “How do I conduct an evaluation?”
Chat History:


Q1: “What are the key risks identified in the analysis?”
Q2: “How can these be mitigated?”
Latest Question:


“What are the next steps?”
Reformulated Question:
“What are the next steps for mitigating the key risks identified in the analysis?”

<|eot_id|><|start_header_id|>user<|end_user_id|>
chat history: {chat_history},
user question: {user_question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["chat_history", "user_question"]
)

cojoin_chain=cojoin_question|llm| StrOutputParser()

docsearch=FAISS.load_local(folder_path="D:\Document\django-chatbot\django_chatbot\website_faiss",embeddings=hf, allow_dangerous_deserialization=True)

retriever=MultiQueryRetriever.from_llm(retriever=docsearch.as_retriever(search_type='similarity',search_kwargs={'k':3, 'fetch_k':10}), llm=llm)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": rag_prompt}
)
groq_llm=ChatGroq(model="llama3-70b-8192", temperature=0)

df=pd.read_csv("D:\Document\django-chatbot\django_chatbot\\new_df.csv")

custom_prompt = """
You are a helpful data analysis assistant. Use the given dataset to answer questions.
Provide detailed responses and explain your reasoning if necessary.
"""

pandas_agent = create_pandas_dataframe_agent(
    llm=groq_llm,
    df=df,
    verbose=False,
    system_message=custom_prompt,
    allow_dangerous_code=True
)

chat_history=[]
def generate_response(message):
    context=""
    qa_ans=qa_chain({"question": message})
    qa_ans=qa_ans["answer"]
    context+=qa_ans
    agent_question=cojoin_chain.invoke({"chat_history":chat_history, "user_question":message})
    try:
       agent_response = pandas_agent.invoke({"input": agent_question})
       context+=agent_response
    except:
       pass
    response=rag_chain.invoke({"question":message, "context":context})
    return response

# Chatbot view
def chatbot(request):
    # Retrieve the chat history from the session (or an empty list if not available)
    chats = request.session.get('chats', [])

    if request.method == 'POST':
        message = request.POST.get('message')
        response = generate_response(message)  # Generate a dummy response

        # Store the chat history in the session
        chat = {'message': message, 'response': response, 'created_at': timezone.now().strftime('%Y-%m-%d %H:%M:%S')}
        chats.append(chat)
        request.session['chats'] = chats  # Save updated chat history in the session

        return JsonResponse({'message': message, 'response': response, 'created_at': chat['created_at']})

    return render(request, 'chatbot.html', {'chats': chats})
