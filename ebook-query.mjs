import 'dotenv/config';
import {
    MilvusClient,
    DataType,
    MetricType,
    IndexType,
} from '@zilliz/milvus2-sdk-node';
import {
    OpenAIEmbeddings
} from '@langchain/openai';

const ADDRESS = process.env.MILVUS_ADDRESS;
const TOKEN = process.env.MILVUS_TOKEN;
const VECTION_DIM= 1024;
const COLLECTION_NAME = 'ebook';

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

async function main() {
    try {
       console.log('Connection to Milvus...');
        await client.connectPromise;
        try {
            await client.loadCollection({
                collection_name: COLLECTION_NAME,
            })
        } catch(err) {
            console.log('Collection already loaded');
        }
        const query = '段誉会什么武功？';
        const queryVector = await getEmbedding(query);
        const searchResult = await client.search({
            collection_name: COLLECTION_NAME,
            vector: queryVector,
            limit: 3,
            metric_type: MetricType.COSINE,
            output_fields: ['id','content','book_id','chapter_num','index','book_name'],
        })

        searchResult.results.forEach((item, index) => {
            // toFixed(2) 保留2位小数
            console.log(`\n 第 ${index + 1} 个结果: Score: ${item.score.toFixed(2)}`);
            console.log(`ID: ${item.id}`);
            console.log(`Content: ${item.content}`);
            console.log(`Book ID: ${item.book_id}`);
            console.log(`Chapter Number: ${item.chapter_num}`);
            console.log(`Index: ${item.index}`);
            console.log(`Book Name: ${item.book_name}`);
        })

    } catch(err) {
        console.error('Connection to Milvus failed:', err.message);
        throw err;
    }
}

main();