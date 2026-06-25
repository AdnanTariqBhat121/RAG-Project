from langchain_community.document_loaders import WebBaseLoader
import os
os.environ["USER_AGENT"] = "my-rag-app/1.0"
url = "https://www.apple.com/in/macbook-pro/?afid=p240%7Cgo~cmp-11182149775~adg-109263622253~ad-784581523344_kwd-987394769~dev-c~ext-~prd-~mca-~nt-search&cid=aos-in-kwgo-txt-mac-mac--"

data = WebBaseLoader(url)
docs = data.load()
print(docs[0].page_content)