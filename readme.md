# RAG 电子书

- 一本电子书，如何做rag

- RAG 的流程
1. 知识库
2. @langchain/community
    来自社区的各种loader
3. Splitter
4. Document
    pageContent
    meta:
5. Embedding Model
6. Milvus

## 开发流程
- ensureBookCollection
    - 判断集合是否存在 hasCollection
    - 创建集合 createConnection 
        schema 
    - 创建索引
    - 加载集合 loadCollection

## MVP
- Vibe Coding
    - 代码平权
    - idea 设计师等
    Minimum Viable Product 最小可执行产品
    cursor/claude code 编程Agent MVP
    产品原型是产品经理设计出来的原型稿
- 正式的商业级别开发
    程序员 继续vibe coding
- 语义搜索和文本匹配
    - 文本匹配 低级搜索  like 模糊搜索 %段誉%
    - 语义搜索更强大