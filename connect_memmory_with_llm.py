import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1 : Setup LLM(Mistral AI with huggingface)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(HUGGINGFACE_REPO_ID):
    llm=HuggingFaceEndpoint(repo_id=HUGGINGFACE_REPO_ID,
                            temperature=0.5,
                            task="text-generation",
                            model_kwargs={"token":HF_TOKEN,
                                          "max_length": 512})
                            
    return llm


# Step 2 : Connect huggingface with FAISS and create chain
DB_FAISS_PATH="vectorstore/db_faiss"

if not os.path.exists(DB_FAISS_PATH):
    raise FileNotFoundError(f"FAISS database directory not found at {DB_FAISS_PATH}")

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside of the given context.

Context: {context}  # Relevant info extracted will go here
Question: {question}  # The user's question is here

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_promt_template):
    prompt=PromptTemplate(template=custom_promt_template,input_variables=["context","question"])
    return prompt

#Load Database
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain 
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
)

#Now invoke chain with  a single query

user_query=input("Write query here: ")
response=qa_chain.invoke({'query':user_query})
print("RESULT:",response['result'])
print("SOURCE DOCUMENTS:",response["source_documents"])

