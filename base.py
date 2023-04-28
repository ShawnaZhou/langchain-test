from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

# 加载文件
loader = UnstructuredPDFLoader("./test.pdf", mode="elements")
data = loader.load()
print('dataLoaded', data)
# 建立向量索引
index = VectorstoreIndexCreator().from_loaders([loader])

while True:
    query = input("Q：")
    result = index.query_with_sources(query)
    print('A：', result)
