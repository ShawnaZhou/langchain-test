from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from textsplitter import ChineseTextSplitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os

# 加载文件
file_path = os.path.abspath("/home/dev/model_inputs/retry/test.pdf")
loader = file_path.endswith(".pdf") and PyPDFLoader(file_path) or TextLoader(file_path)

# 定义文本分块的规则，这里用了一个很简单的规则，按照默认的分隔符来切割文本，使得每一段不超过1000个字符
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
splitter = CharacterTextSplitter(separator="\n",
                                 chunk_size=200,
                                 chunk_overlap=200,
                                 )
# splitter = ChineseTextSplitter(pdf=True)
chunks = loader.load_and_split(splitter)

# 把文本转换为向量。
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
# 建立向量索引
indexStore = Chroma.from_documents(chunks, embeddings)
# 建立向量搜索器
retriever = indexStore.as_retriever(search_kwargs={"k": 5})

while True:
    query = input("Q：")
    result = retriever.get_relevant_documents(query)
    # for content in result
    print('A', result[0].page_content, '\n', result[1].page_content, '\n', result[2].page_content, '\n',
          result[3].page_content)
