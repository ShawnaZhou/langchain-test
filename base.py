from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from textsplitter import ChineseTextSplitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os

# 加载文件
file_path = os.path.abspath("/home/dev/model_inputs/retry/test.pdf")
loader = UnstructuredPDFLoader(file_path, mode="elements")

# 定义文本分块的规则，这里用了一个很简单的规则，按照默认的分隔符来切割文本，使得每一段不超过1000个字符
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# splitter = CharacterTextSplitter(separator="\n\n",
#                                  chunk_size=1000,
#                                  chunk_overlap=200,
#                                  length_function=len,
#                                  )
textsplitter = ChineseTextSplitter(pdf=True)
chunks = loader.load_and_split(textsplitter)

# 把文本转换为向量。
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
# 建立向量索引
indexStore = Chroma.from_documents(chunks, embeddings)
retriever = indexStore.as_retriever()

while True:
    query = input("Q：")
    result = retriever.get_relevant_documents(query)
    # for content in result
    print('A', result[0].page_content, result[1].page_content, result[2].page_content, result[3].page_content)
