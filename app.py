import os
import re
import uuid

from dotenv import load_dotenv

load_dotenv()

import getpass
import time
from typing import Optional, Sequence

import streamlit as st
from langchain import hub
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import messages_to_dict
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    MessagesPlaceholder)
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from PyPDF2 import PdfReader

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, trim_messages)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    )  
template = """Answer the following question based on the provided context
                <context>
                {context}
                </context>

                Question:{input}
            """


class State(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  language: str
  context: str

#define Graph
workflow = StateGraph(state_schema=State)
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
            " Be concise and to the point and dont be nice or acknowledge the question you'll also optionally get a {context} to answer questions off of."
            " Also make sure to tell me what part of the context you got you answer from Output this in a new line with Source(if the answer has a source):. Please make sure to add 2 line breaks before the source"
            " If the answer is not in the previous messages or context say that it doesnt make sense in this context but "
            "You are allowed to be polite and respond to greetings"
            "Be extremely mad if you get asked the same question 2 times.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)



def call_model(state:State):
  
  prompt = prompt_template.invoke({
    "messages": state["messages"],
    "language": state["language"],
    "context": state["context"],})
  response = model.invoke(prompt)
  return {"messages":response, "language":state["language"],"context": state["context"]}

#define single node of a graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

#add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    #st.write(text)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore= FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def display_messages():
     # Display all past messages
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "ai"
        with st.chat_message(role):
            st.markdown(msg.content)


def handle_user_input(question, vectorstore:FAISS):
    relevant_embeddings = vectorstore.similarity_search(query=question, k=5)
    context = "\n\n".join(doc.page_content for doc in relevant_embeddings)
    if "language" not in st.session_state:
        st.session_state.language = "Scientific and concise english" 

    config = {"configurable": {"thread_id": "abc457"}}
    query = question
    language = st.session_state.language
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if question.startswith("/language "):
        # Extract the desired language setting
        new_language = question[len("/language "):].strip()
        if new_language == "reset":
            st.session_state.language = "Scientific and concise english"
            st.success("Language reset to clear and concise english.")
            time.sleep(1)
            display_messages()   
            return
        st.session_state.language = new_language
        st.success(f"Language updated to: {new_language}")
        time.sleep(1)
        display_messages()
        return

    if question == "/clear":
        st.session_state.messages.clear()
        st.success("Chat History Cleared!")
        return
    
    display_messages()    
    

    with st.chat_message("user"):
        st.markdown(question)
    
    
        
    st.session_state.messages.append(HumanMessage(query))

    input_messages = st.session_state.messages
    output = app.invoke({"messages": input_messages,
                        "language": language,
                        "context": context
                        }, config)
    ai_message = output["messages"][-1]
    st.session_state.messages.append(ai_message)
    with st.chat_message("ai"):
        st.markdown(ai_message.content)
        
def main():
    

    st.set_page_config(page_title="PDF ChatBOT", page_icon=":battery:")

    
    st.header("PDF ChatBOT:battery:")
    question = st.chat_input("Ask anything")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("upload your PDFs here and click process.", accept_multiple_files = True)
        
        if st.button("process") and pdf_docs:
            start = time.perf_counter()
            with st.spinner("Parsing Text"):                    
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                text_parsed = time.perf_counter()
                st.write(f"Text parsed in {(text_parsed-start):.2f}s")

            with st.spinner("Splitting Text into chunks"):
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                text_chunked = time.perf_counter()
                st.write(f"Text split into chunks in {(text_chunked-text_parsed):.4f}s")     

            with st.spinner("Embedding"):           
                #create vector store
                st.session_state.vectorstore = get_vector_store(text_chunks)    
                
                embedding_time = time.perf_counter()
                st.write(f"Embedding Complete in {(embedding_time-text_chunked):.2f}s")

            with st.spinner("Creating Summary"):    
                #create and print summary
                load_dotenv()
                       
                config = {"configurable": {"thread_id": "summary"}}
                input_messages = [HumanMessage(content="Ignore earlier instructions and only give the sumary of this text in 50 words or less:" + raw_text)]
                language = "Clear and Concise scientific English"
                context = ""
                summary = app.invoke(
                    {"messages": input_messages, "language": language, "context": context},
                    config,
                    )
                
                
                summary_time = time.perf_counter()
                st.write(f"Summary completed in {(summary_time-embedding_time):.2f}s")
                st.write(f"Processing complete in {(summary_time-start):.2f}s")
                tldr = summary["messages"][-1].content
                st.write(tldr)
                
                

    if question and st.session_state.vectorstore:
        vectorstore = st.session_state.vectorstore
        handle_user_input(question, vectorstore)

    if question and st.session_state.vectorstore == None:
        with st.chat_message("system"):
            st.markdown("Upload and Process Documents to start chatting.") 

               
                

        


if __name__ == "__main__":
    main()
