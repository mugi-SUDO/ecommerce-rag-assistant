# data/processed

存放中间处理产物（可选），例如分块后的文本缓存、向量索引等。
当前实现：向量索引持久化在项目根目录 `vector_index/`（见 config.chroma_persist_dir），
本目录预留用于后续扩展离线预处理脚本输出。
