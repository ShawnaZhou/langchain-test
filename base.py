from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

# 加载文件
file_path = os.path.abspath("/home/dev/model_inputs/retry/test.pdf")
loader = UnstructuredPDFLoader(file_path, mode="elements")
data = loader.load()
print('dataLoaded', data)
# 建立向量索引
indexStore = VectorstoreIndexCreator().from_loaders([loader])

while True:
    query = inputStore("Q：")
    result = indexStore.query_with_sources(query)
    print('A：', result)
