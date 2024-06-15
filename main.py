import os, sys, time, pickle
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, abort, send_file
 
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    

os.environ["NVIDIA_API_KEY"]  = "****Enter Your NVIDIA_API_KEY****"


app = Flask(__name__)

line_bot_api = LineBotApi('*********************')
handler = WebhookHandler('*********************')


#######################################################################
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

llm = ChatOpenAI(
            model="meta/llama3-70b-instruct",
            openai_api_key=os.environ["NVIDIA_API_KEY"],
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=0.0,
            max_tokens=1024,
            model_kwargs={"top_p": 1},
        )
document_embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

prompt = hub.pull("hwchase17/react-chat")

#######################################################################
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import BaseTool
from langchain_core.documents import Document
from typing import List

class MyWebLoader(WebBaseLoader):
    """Load My webpages."""

    def load(self) -> List[Document]:
        """Load webpages."""
        docs = []
        for path in self.web_paths:
            print(path)
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            try:
                text = soup.select_one("div[class*='main_content']").text
            except:
                text = soup.select_one("div[id='printarea']").text
            metadata = {"source": path}
            docs += [Document(page_content=text, metadata=metadata)]
        return docs
    
# Path to the vector store file
DOCS_DIR = "./documents/"
vector_store_path = "vectorstore.pkl"

web_list = ["https://www.cmuh.cmu.edu.tw/Department/Detail?depid=96", #科室首頁
            "https://www.cmuh.cmu.edu.tw/Department/NewsInfo?depid=96", #科室訊息
            "https://www.cmuh.cmu.edu.tw/Department/AboutUs?depid=96", #認識我們
            "https://www.cmuh.cmu.edu.tw/Department/Advanced?depid=96", #引領卓越
            "https://www.cmuh.cmu.edu.tw/Department/Feature?depid=96", #特色介紹
            "https://www.cmuh.cmu.edu.tw/Department/History?depid=96", #大事記
            "https://www.cmuh.cmu.edu.tw/Department/CustomPageList/208", #學術研討
            "https://www.cmuh.cmu.edu.tw/Department/Team?detail=96&current=13&source=dep", #團隊介紹
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D4615",  #梁基安
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D16181", #簡君儒
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D4634",  #陳尚文 
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D17473", #王耀慶
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D13464", #林膺峻
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D14360", #朱俊男
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D22687", #賴宥良
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D30860", #王帝皓
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D31968", #林亭君
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D14069", #廖志穎
            "https://www.cmuh.cmu.edu.tw/Doctor/DoctorInfo?docId=D28438",  #黃繼賢
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4956", #放射線治療簡介
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4955", #放射線治療常見疑慮澄清
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4882", #頭頸部放射治療口腔併發症之預防
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4958", #放射線治療與牙關緊閉
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=5614", #放射線治療與頸部肌肉纖維化
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=5476", #放射性皮膚炎之預防
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=5699", #放射線治療的皮膚照顧
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4949", #放射線治療與味覺改變
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4952", #放射線治療後之陰道沖洗器使用
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4957", #放射線治療與腹瀉     
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4948", #放射線治療與性生活            
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4947", #放射線治療與皮膚炎            
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4954", #放射線治療後之陰道部位復健
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4951", #放射線治療與口腔黏膜炎           
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=5562", #放射線治療中醫護理           
            "https://www.cmuh.org.tw/HealthEdus/Detail?no=4950", #止痛吩坦尼貼片用藥指導
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=4794", #21 癌症治療期間我該怎麼吃？
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=4795", #22 癌症副作用飲食(1)
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=4793", #23 癌症副作用飲食(2)
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=5540", #24 粉狀營養品沖泡說明
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=4783", #25 全胃切除傾食症候群
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=5712", #26 兒童癌症治療飲食
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=5679", #27 癌症末期飲食照護
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=5678", #28 居家管灌方法與注意事項
            "https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=6384", #29 康復期健康餐盤            
           ]


embedding_model = "intfloat/multilingual-e5-large"

if os.path.exists(vector_store_path)==False:
    
    # Load raw documents from the directory
    loader = MyWebLoader(web_list)
    raw_documents = loader.load()

    # Check for existing vector store file
    if raw_documents:
        text_splitter = SentenceTransformersTokenTextSplitter(model_name=embedding_model, tokens_per_chunk=512, chunk_overlap=128)
        documents = text_splitter.split_documents(raw_documents)

        print("Adding document chunks to vector database...")
        vectorstore = FAISS.from_documents(documents, document_embedder)

        print("Saving vector store")
        try:
            with open(vector_store_path, "wb") as f:
                pickle.dump(vectorstore, f)
            print("Vector store created and saved.")
        except:
            print("No documents available to process!")
else:
    vectorstore = None
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
        
        
        
def _get_RT_llm(llm) -> BaseTool:

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})
    
    return Tool(
        name="RT Assistant",
        description="Useful for when you need to answer questions about China Medical University Hospital. Input should be a single string strictly.",
        
        func=RetrievalQA.from_llm(
            llm = llm, 
            retriever = retriever,
            return_source_documents=False
        ).run,
        
        coroutine=RetrievalQA.from_llm(
            llm = llm, 
            retriever = retriever,
            return_source_documents=False
        ).arun,
        
    )        
        
#######################################################################   
@app.route("/", methods=['POST'])
def index():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
 
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
 
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
 
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    fDEBUG     = False
    fSearch    = False
    profile    = line_bot_api.get_profile(event.source.user_id)
    user_id    = profile.user_id    
    input_text = str(event.message.text).strip()
    inputStrs  = input_text.split(" ")
    print(f"\n{bcolors.OKCYAN}Name: {profile.display_name} - ID: {user_id} \nURL: {profile.picture_url} \nInput: {input_text} {bcolors.ENDC}") 
    memory_path = f"./records/{user_id}.pkl"   


    if (input_text.upper()=="HELP"):
        show_text = '請直接輸入和放射腫瘤部門相關問題進行諮詢。' 
        line_bot_api.reply_message(event.reply_token,
                                   TextSendMessage(show_text))
        
    elif (input_text.upper()=="CLEAR"):
        if os.path.exists(memory_path): os.remove(memory_path)
        line_bot_api.reply_message(event.reply_token,
                                   TextSendMessage("已清除暫存對話資訊。"))
    
    else:
        try:
            print(f"chat: {memory.chat_memory.messages}")
        except:
            pass
        
        if fDEBUG: print("Processing Message")

        if os.path.exists(memory_path):
            mtime_memory = os.path.getmtime(memory_path)
            lifecycle = time.time() - mtime_memory
            if lifecycle < 3600:                
                with open(memory_path, 'rb') as file:
                    memory = pickle.load(file)
            else:
                memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)
        else:
            memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)
            
        chat_len = len(memory.chat_memory.messages)
            
        if fDEBUG: print(f"Memory: {memory.chat_history}")
        
        try:
            prompt.template='現在時間是{timestamp}，你是中國醫藥大學附設醫院放射腫瘤部門的專業助手，名叫RT Agent。中國醫藥大學附設醫院的放射腫瘤部的地址是：40447台中市北區育德路2號。連絡諮詢專線：(04)22052121分機17450、17454。若你被詢問任何醫療人員的資訊或是評價問題，請他到團隊介紹網頁查詢 (https://www.cmuh.cmu.edu.tw/Department/Team?detail=96&current=13&source=dep)。作為一個專業助理，你能夠根據接收到的輸入生成類似人類的文本，使其能夠進行自然流暢的對話，並提供與主題相關且連貫的詳盡回應，從而參與討論並提供關於各種主題的解釋和描述。你將根據上下文資訊回答放射治療的相關問題。如果上下文內容和問題不相關，請避免回答並禮貌地拒絕回答用戶，並建議他直接洽詢中國醫藥大學附設醫院的放射腫瘤部。使用者可能有情緒低落的問題，因此你的所有回應請嚴謹應對。總而言之，你是一個強大的繁體中文智慧助理，可以完成各種任務，並在各種主題上提供有價值的見解和詳盡的資訊。若您被詢問有關放射腫瘤相關的衛生教育問題請使用RT_Assistant工具回答。\n\nTOOLS:\n------\n\n你具有使用下列工具的權限:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nFinal Answer: [your response here in Chinese and make it fluence]\n```\n\nBegin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}'
            
            RT_Assistant = _get_RT_llm(llm)
            tool_names =[]
            tools=load_tools(tool_names, llm)
            tools.append(RT_Assistant)
            
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent, tools=tools, memory=memory, verbose=True, max_iterations=10, handle_parsing_errors=True
            )


            response = agent_executor.invoke({"input": input_text, \
                                              "question": input_text, \
                                              "timestamp": datetime.now().strftime("%Y年%m月%d日 %H點%M分%S秒")})
            
            
            chat_response = (f'({chat_len//2+1}/10)\n{response["output"]}')
            
            line_bot_api.reply_message(event.reply_token,
                                       TextSendMessage(chat_response))
            
            print(f"{bcolors.OKGREEN}ChatBot: {chat_response}{bcolors.ENDC}")
            
            if (chat_len+2)//2 == 10:
                memory.chat_memory.clear()
                if os.path.exists(memory_path): os.remove(memory_path)
            else:        
                with open(memory_path, 'wb') as file:
                # Serialize and write the variable to the file
                    pickle.dump(memory, file)
                    
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            
            memory.chat_memory.clear()
            if os.path.exists(memory_path): os.remove(memory_path)
            line_bot_api.reply_message(event.reply_token,
                                       TextSendMessage(f"請重新提問。\n{e}"))

        
        
if __name__ == "__main__":
    app.run(port=5002)     
    