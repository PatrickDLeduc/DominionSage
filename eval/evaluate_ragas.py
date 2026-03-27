import os
import sys

# Add project root to path so we can import the retrieval package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from retrieval.orchestrator import answer_question
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Define a small evaluation dataset
EVAL_DATASET = [
    {
        "question": "When can I play a Reaction card?",
        "ground_truth": "You can play a Reaction card when its specific trigger condition is met, such as when another player plays an Attack card (like Moat).",
    },
    {
        "question": "What happens when the Supply runs out?",
        "ground_truth": "The game ends immediately at the end of a player's turn if either the Province pile is empty, or any 3 Supply piles are empty.",
    },
    {
        "question": "How do Duration cards work?",
        "ground_truth": "Duration cards stay in play until the Clean-up phase of the turn they are finished doing things (usually the next turn). They have effects that happen on future turns.",
    },
    {
        "question": "What does Chapel do?",
        "ground_truth": "Chapel allows you to trash up to 4 cards from your hand.",
    }
]

def main():
    load_dotenv()
    
    print("Preparing RAGAS evaluation dataset...")
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in EVAL_DATASET:
        question = item["question"]
        print(f"Running pipeline for question: '{question}'")
        
        # Query our RAG pipeline
        result = answer_question(question, limit_cards=True)
        answer = result["answer"]
        
        # Extract raw text from the matched sources to serve as contexts
        contexts = []
        for source in result["sources"]:
            if source["type"] == "rulebook":
                contexts.append(source["data"].get("chunk_text", ""))
            elif source["type"] == "card_db":
                card = source["data"]
                # Formulate a text chunk representing the card stats
                card_text = f"Card Name: {card.get('name', '')} | Cost: {card.get('cost', '?')} | Type: {card.get('type', '?')} | Text: {card.get('text', 'N/A')}"
                contexts.append(card_text)
                
        # RAGAS requires lists for each column
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(item["ground_truth"])

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(data)

    print("\nRunning RAGAS evaluation...")
    
    # explicit models
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    # We pass the metrics properly initialized with the explicit models.
    
    # Using existing OpenAI API keys from environment
    eval_result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    
    print("\n=== RAGAS Evaluation Results ===")
    print(eval_result)
    
    # Export the granular results to CSV
    df = eval_result.to_pandas()
    os.makedirs("eval", exist_ok=True)
    df.to_csv("eval/ragas_results.csv", index=False)
    print("\nDetailed results saved to eval/ragas_results.csv")

if __name__ == "__main__":
    main()
