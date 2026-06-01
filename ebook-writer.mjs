import 'dotenv/config';
import { parse } from 'path';
import {
    MilvusClient,
    DataType,
    MetricType,
    IndexType,
} from '@zilliz/milvus2-sdk-node';
import {
    OpenAIEmbeddings
} from '@langchain/openai';
import {
    EPubLoader // 加载EPUB文件
} from '@langchain/community/document_loaders/fs/epub';
import {
    RecursiveCharacterTextSplitter
} from '@langchain/textsplitters';

const COLLECTION_NAME = 'ebook';
const VECTION_DIM= 1024;
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 50;
const EPUB_FILE = './天龙八部.epub';

const ADDRESS = process.env.MILVUS_ADDRESS;
const TOKEN = process.env.MILVUS_TOKEN;

const BOOK_NAME = parse(EPUB_FILE).name;
// console.log(BOOK_NAME);

const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
    dimensions: VECTION_DIM,
})

const client = new MilvusClient({
    address: ADDRESS,
    token: TOKEN,
    ssl: true,
})

async function getEmbedding(text) {
    const result = await embeddings.embedQuery(text);
    return result;
}

// 确保电子书集合存在, 如果不存在则创建集合
async function ensureBookCollection(bookId) {
    try {
        const hasCollection = await client.hasCollection({
            collection_name: COLLECTION_NAME,
        })
        console.log('集合是否存在:', hasCollection);
        if(!hasCollection.value) {
            console.log(`${COLLECTION_NAME} 集合不存在，创建集合`);
            await client.createCollection({
                collection_name: COLLECTION_NAME,
                // schema
                fields: [
                    { name: 'id', data_type: DataType.VarChar, max_length: 100, is_primary_key: true },
                    { name: 'book_id', data_type: DataType.VarChar, max_length: 100 },
                    { name: 'book_name', data_type: DataType.VarChar, max_length: 100 },
                    { name: 'chapter_num', data_type: DataType.Int32 },
                    { name: 'index', data_type: DataType.Int32 },
                    { name: 'content', data_type: DataType.VarChar, max_length: 10000 },
                    { name: 'vector', data_type: DataType.FloatVector, dim: VECTION_DIM },
                ]
            });
            console.log('集合创建成功');
            await client.createIndex({
                collection_name: COLLECTION_NAME,
                field_name: 'vector',
                // IVF_FLAT 索引类型 倒排文件+平面存储 精度接近暴力搜索 速度较快
                index_type: IndexType.IVF_FLAT, 
                metric_type: MetricType.COSINE,
                params: {
                    nlist: VECTION_DIM,
                }
            });
            console.log('索引创建成功');
        }
        // 已经加载集合, 则直接返回
        try {
            await client.loadCollection({
                collection_name: COLLECTION_NAME,
            });
            console.log('集合加载成功');
        } catch(err) {
            console.log('集合已处于加载状态');
        }
    } catch(err) {
        console.error('创建集合失败:', err.message);
        throw err;
    }
}

// 加载EPUB文件, 并对内容进行拆分, 并将结果写入Milvus
async function loadAndProcessEPubStreaming(bookId) {
    try {
        console.log('开始加载EPUB文件');
        const loader = new EPubLoader(
            EPUB_FILE,
            {
                splitChapters: true,
            }
        );
        const documents = await loader.load();
        console.log(documents);
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: CHUNK_SIZE,
            chunkOverlap: CHUNK_OVERLAP,
            // 默认 \n\n 分割 两个是段落的换行 一个\n是换行符
        })
        let totalInserted = 0;
        for (let chapterIndex = 0; chapterIndex < documents.length; chapterIndex++) {
            const chapter = documents[chapterIndex];
            const chapterContent = chapter.pageContent;
            console.log(`处理第${chapterIndex + 1}/${documents.length} 章`);
            const chunks = await textSplitter.splitText(chapterContent);
            console.log(`拆分为 ${chunks.length} 个片段`);
            if(chunks.length === 0) {
                console.log(`跳过空章节\n`);
                continue;
            }
            console.log('生成向量并插入中...');
            const insertedCount = await insertChunksBatch(chunks, bookId, chapterIndex + 1);
            totalInserted += insertedCount;
            console.log(`已插入 ${insertedCount} 个片段, 累计插入${totalInserted} 个片段`);
        }
        console.log(`\n处理完成 ，共插入 ${totalInserted} 个片段`);
        return totalInserted;
    } catch(err) {
        console.error('加载EPUB文件失败:', err.message);
        throw err;
    }
}

async function insertChunksBatch(chunks, bookId, chapterNum) {
    try {
        if(chunks.length === 0) {
            return 0;
        }
        // 性能优化 embedding 并发 
        // 返回结果是符合 schema 的数组
        const insertData = await Promise.all(
            chunks.map(async (chunk, chunkIndex) => {
                const vector = await getEmbedding(chunk);
                return {
                    id: `${bookId}_${chapterNum}_${chunkIndex}`,
                    book_id: bookId,
                    book_name: BOOK_NAME,
                    chapter_num: chapterNum,
                    index: chunkIndex,
                    content: chunk,
                    vector
                }
            })
        )
        // 批量插入
        const insertResult = await client.insert({
            collection_name: COLLECTION_NAME,
            data: insertData,
        })
        return Number(insertResult.insert_cnt) || 0;
    } catch(err) {
        console.error('插入片段失败:', err.message);
        throw err;
    }
}

// 主函数, 处理电子书, 并将内容写入Milvus
async function main() {
    try {
        console.log('电子书处理');
        console.log('连接Milvus');
        await client.connectPromise;
        console.log('连接Milvus成功');

        const bookId = 1;
        await ensureBookCollection(bookId); // 确保电子书集合存在
        await loadAndProcessEPubStreaming(bookId);
    } catch(err) {
        console.error('执行失败:', err.message);
    }
}

main();