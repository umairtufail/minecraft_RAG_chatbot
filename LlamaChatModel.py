from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
import chainlit as cl
import textwrap
from langchain import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
DB_FAISS_PATH = './vectorstore/db_faiss'
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    # print('\n\nSources:')
    # for source in llm_response["source_documents"]:
    #     print(source.metadata['source'])
custom_prompt_template = """
    Fungiere als Minecraft-Support-Assistent.
    Beantworten Sie die Frage anhand des gegebenen Kontexts zum Thema so gut Sie können.
    Context: {context}
    Question:
    {question}
    Answer:
    """
   

def SetCustomPrompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def RetrievalQAChain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 1}), 
                                           return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain
model_path= "C://Users//umair//Documents//GitHub//minecraft_chatbot//model//llama-2-7b-chat.Q4_K_S.gguf"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
def LoadLLM():
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=40,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
    )

    return llm

def LLamaQABot():
    # Load LLM model
    llm = LoadLLM()

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Load FAISS vector store
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    # Set custom prompt
    qaPrompt = SetCustomPrompt()

    # Create retrieval QA chain
    qaBot = RetrievalQAChain(llm, qaPrompt, db)

    return qaBot


def GetAnswer(query):
    qaBot = LLamaQABot()
    response = qaBot({'query':query})
    process_llm_response(response)
    return response

@cl.on_chat_start
async def Start():
    chain=LLamaQABot()
    message = cl.Message(content="Starting Minecraft Bot...")
    await message.send()
    message.content = "Hi, Welcome to Minecraft Chat Bot. How may I help you?"
    await message.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    langchainCallBack = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])

    langchainCallBack.answer_reached=True
    response = await chain.acall(message.content, callbacks=[langchainCallBack])
    answer = response["result"]
    sources = response["source_documents"]

    # if sources:
    #     sources += f"\nSources:" + str(sources)
    # else:
    #     sources += "\nNo sources found"

    await cl.Message(content=answer).send()









    