from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
# 加载文件
file_path = os.path.abspath("/home/dev/model_inputs/retry/test.pdf")
loader = UnstructuredPDFLoader(file_path, mode="elements")

# 定义文本分块的规则，这里用了一个很简单的规则，按照默认的分隔符来切割文本，使得每一段不超过1000个字符
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = loader.load_and_split(splitter)

# 把文本转换为向量。
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# 建立向量索引
indexStore = VectorstoreIndexCreator().from_documents(chunks, embeddings)

while True:
    query = inputStore("Q：")
    result = indexStore.query_with_sources(query)
    print('A：', result)
