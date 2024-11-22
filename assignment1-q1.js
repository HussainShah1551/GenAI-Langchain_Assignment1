import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import * as dotenv from "dotenv";
import { FireworksEmbeddings } from "@langchain/community/embeddings/fireworks";
import {ChatPromptTemplate, MessagesPlaceholder} from "@langchain/core/prompts"

import readline from "readline";

import { AIMessage , HumanMessage } from "@langchain/core/messages";

dotenv.config();

const embeddings = new FireworksEmbeddings();

const model = new ChatFireworks({
    model: "accounts/fireworks/models/mixtral-8x7b-instruct",
    base_url: "https://api.fireworks.ai/inference/v1/completions",
    apiKey :process.env.FIREWORKS_API_KEY,
    temperature: 0.4,
});










const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history")

    ,['user' , '{input}'],
    ['user','You are a movie addict, an expert in explaining the plots  of different movies and answering questions related to them. For any movie mentioned, provide a  plot summary. Do not answer questions unrelated to movies and apologize politely and briefly  that you are unable to answer. If a movie is not recognized, say so and do not make up information.Here is your Question: {input}']

])





const chain = await retrieverPrompt.pipe(model)


//Initializing Chat History

let chatHistory = [];



const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

//Function to  have a Conversation with the user based on previous history and also keep the convo running  until user enter the Word "End"


async function askQuestion() {

console.log("\n\nStart by asking a question  (You can End the chat by entering 'End' as your response\n\n ")






    rl.question("🤖: Hi, lets talk about movies  ! How can  I help you?\n\n❓:", async (user_input) => {
        if (user_input === "End") {
            console.log("Goodbye! Let's talk about movies some other day.");
            
            rl.close();
        } else {
            
            chatHistory.push(new HumanMessage(user_input));

            console.log("\n\n👤:" , user_input)

            const karen_response = await chain.invoke({
                input: user_input,
                chat_history : chatHistory   
            
            })

            
            const aiResponse = karen_response.content;
            console.log("\n\n🤖: " , aiResponse)
            chatHistory.push(new AIMessage(aiResponse));

            
            askQuestion(); 

        }
    });
}
askQuestion();





