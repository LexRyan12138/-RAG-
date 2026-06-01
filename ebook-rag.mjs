import 'dotenv/config';
import {
    MilvusClient,
    DataType,
    MetricType,
    IndexType,
} from '@zilliz/milvus2-sdk-node'
import {
    ChatOpenAI,
    OpenAIEmbeddings
} from '@langchain/openai'


const ADDRESS = process.env.MILVUS_ADDRESS;
const TOKEN = process.env.MILVUS_TOKEN;
const COLLECTION_NAME = 'ebook';
const VECTION_DIM = 1024;

const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: process.env.EMBEDDING_MODEL_NAME,
    configuration:{
        baseURL: process.env.OPENAI_BASE_URL,
    },
    dimensions: VECTION_DIM,
})

const client = new MilvusClient({
    address: ADDRESS,
    token: TOKEN,
})

const model = new ChatOpenAI({
    temperature:0.7,
    model:process.env.MODEL_NAME,
    apiKey:process.env.OPENAI_API_KEY,
    configuration:{
        baseURL:process.env.OPENAI_BASE_URL,
    }
})

async function getEmbedding(text) {
    const result = await embeddings.embedQuery(text);
    return result;
}

// 检索与问题最相关的的内容
async function retrievedRelevantContent(question, k=3) {
    try {
        const queryVector = await getEmbedding(question);

        const searchResult = await client.search({
            collection_name: COLLECTION_NAME,
            vector: queryVector,
            limit: k,
            metric_type: MetricType.COSINE,
            output_fields: ['id','book_id','book_name','chapter_num','index','content']
        });
        return searchResult.results;
    } catch(err) {
        console.error('向量搜索失败:', err.message);
        return [];
    }
}

// 回答问题
async function answerEbookQuestion(question, k=3) {
    try {
        console.log('开始回答问题: question');
        
        const retrievedContent = await retrievedRelevantContent(question, k);
        console.log(`[检索到的内容]`, retrievedContent);
        if(retrievedContent.length === 0) {
            console.log('未检索到相关内容');
            return '抱歉, 未检索到相关内容';
        }
        
        // 格式化检索到的内容
        const context = retrievedContent
        .map((item, i) => `
            [片段${i+1}]
            章节:第 ${item.chapter_num} 章
            内容: ${item.content}    
        `).join('\n\n----\n\n');

        // 构建prompt
        const prompt = `
        你是一个专业的《天龙八部》小说助手。基于小说内容回答问题，用准确详细的语言。

        请根据以下《天龙八部》小说片段内容回答问题：
        ${context}

        用户问题: ${question}

        回答要求：
        1. 如果片段中有相关信息，请结合小说内容给出详情，准确的回答
        2. 可以综合多个片段的内容，提供完整的答案
        3. 如果片段中没有相关的信息，请如实告知用户
        4. 回答要准确，符合小说的情节和人物设定
        5. 可以引用原文内容来支持你的回答

        AI助手的回答：
        `

        console.log('AI助手的回答');
        
        const response = await model.invoke(prompt);
        console.log(response.content);
        return response.content;

    } catch(err) {
        return '抱歉，处理您的问题时出现了错误'
    }
}



async function main() {
    try {
        console.log('Connection to Milvus...');
        await client.connectPromise;
        try {
            await client.loadCollection({
                collection_name: COLLECTION_NAME,
            });
            console.log('Collection already loaded');
        } catch(err) {
            console.log('Collection not loaded');
        }
        
        const result = await answerEbookQuestion("段誉会什么武功？");
        console.log('最终答案:', result);

    } catch(err) {
        console.error('Connection to Milvus failed:', err.message);
        throw err;
    }
}

main();