import {ChatPromptTemplate, MessagesPlaceholder} from "@langchain/core/prompts"
import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import * as dotenv from "dotenv";
dotenv.config();
import {createStuffDocumentsChain} from "langchain/chains/combine_documents"
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"
import { FireworksEmbeddings } from "@langchain/community/embeddings/fireworks";
import {MemoryVectorStore} from "langchain/vectorstores/memory"
import{createRetrievalChain} from "langchain/chains/retrieval";
import { AIMessage , HumanMessage } from "@langchain/core/messages";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import readline from "readline";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";




// function to create a vector store using Pdf Document containing emumba's loan ploicy 
const createVectorStore = async () => {


const doc_loader = new PDFLoader("C:\\Users\\Emumba\\Desktop\\Personal\\Langchain\\Loan Application.pdf");


const loan_policy = await doc_loader.load();


const splitter = new RecursiveCharacterTextSplitter({
    chunkSize :200 ,
    chunkOverlap : 20 ,
})


const splitted_docs = await splitter.splitDocuments(loan_policy)


const embeddings = new FireworksEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(splitted_docs , embeddings)

return vectorStore

}

//Model Chain 

const create_chain =  async (vectorStore) => {
    const llm_model = new ChatFireworks({
        model: "accounts/fireworks/models/mixtral-8x7b-instruct",
        base_url: "https://api.fireworks.ai/inference/v1/completions",
        apiKey :process.env.FIREWORKS_API_KEY,
        temperature: 0.7,
    });
 
    ;


    //Prompt for Response Generation 
    const rag_prompt = ChatPromptTemplate.fromMessages([
        ["system", 
 "You are an Ai assistant that can only provide info that is present in this  context : {context} ,  you donot know about any other  info but donot mention this limitation  . Donot provide the information on what you can do or  just ask the user to ask you a question"
]   ,
       
        ["user", "{input}"]
    ]);



 //Documents chain 
const chain_rag = await createStuffDocumentsChain({
    llm:llm_model,
    prompt:rag_prompt
    
})

const retriever  = vectorStore.asRetriever({
    k : 10, 

})

const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history")

    ,['user' , '{input}'],
    
   

])




const historyAwareRetriever = await createHistoryAwareRetriever({
    llm : llm_model,
    retriever: retriever,
    rephrasePrompt:retrieverPrompt

})



const retrievalChain = await createRetrievalChain({
    combineDocsChain : chain_rag,
    retriever: historyAwareRetriever
})

return retrievalChain

}

let chain;
//Create Vector Store
(async () => {
    const vectorStore = await createVectorStore();
    chain = await create_chain(vectorStore);

   
})();




//Initializing Chat History

let chatHistory = [];





const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});


//Function to  have a Conversation with the user based on previous history and also keep the convo running  until user enter the Word "End"
async function askQuestion() {




console.log("\n\nStart by asking a question  (You can End the chat by entering 'End' as your response)\n\n ")






    rl.question("🤖: Hi, lets talk about Emumba's Loan Policy  ! How can  I help you?\n❓:", async (user_input) => {

        
        if (user_input === "End") {
            console.log("Nice Talking to You . Bye ");
            
            rl.close();
        } else {
            
            chatHistory.push(new HumanMessage(user_input));

            console.log("\n\n👤:" , user_input)

            const rag_response = await chain.invoke({
                input: user_input,
                chat_history : chatHistory   
            
            })

            
            const aiResponse = rag_response.answer;
            console.log("\n\n🤖: " , aiResponse)
            chatHistory.push(new AIMessage(aiResponse));

            
            askQuestion(); 

        }
    });
}
askQuestion();