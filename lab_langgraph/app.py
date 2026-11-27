from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv.ipython import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

load_dotenv(override=True)
loader = PyPDFLoader("CV -  Alex CHIMI - 20260204.pdf")
tokennizer = tiktoken.encoding_for_model("gpt-4o-mini")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name= tokennizer.name,
    chunk_size = 300,
    chunk_overlap = 20
)

chunks = loader.load_and_split(splitter)
embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="cv_data_collection"
)

retriever = vector_store.as_retriever(kwargs={'k':10})

@tool
def retriever_tool(query : str) -> str:
    """
     Permet de chercher des informations sur des candidats:
     - Nom, Prénom, Diplômes
     - Expériences
     - Competences
    """
    relevant_chunks = retriever.invoke(query)
    context_list = [d.page_content for d in relevant_chunks]
    context = ". ".join(context_list)
    return context

@tool
def get_company_infos(company_name :str):
    """
    Consulter des informations sur l'entreprise donnée
    """
    return {
        "company_name" : company_name,
        "domain" : "IT",
        "turnover": 120_870_000
    }
    
## creer l'agent qui a un RAG comme tools d'abord avec langchain
# agent il faut lui donner le cerveau: llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

my_agent = create_agent(
    model=llm,
    tools=[retriever_tool, get_company_infos], # c'est lui qui permet d'exploiter le RAG
    system_prompt="Répond à la question de l'utilisateur en utilisant les tools fournis"
)
## on peut a ce niveau utiliser langgraph studio afin de tester cet agent
# il faut créer un fichier qui s'appelle langgraph.json
#"agent_rag_graph": "./app.py:agent" <=>"nom du graph: chemin du fichier:variable qui contien agent"
#"env": "./.env" <=>ou se trouve les variables d'environement
# dans la variables d'environement il a besoin de trouver la key OPEN_AI et la key de Langsmith
# uv add "langgraph-cli[inmem]"
#activer env: >source .venv/Scripts/activate
#executer la commande:>langgraph dev
#- 🚀 API: http://127.0.0.1:2024
#- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
#- 📚 API Docs: http://127.0.0.1:2024/docs
