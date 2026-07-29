import os
import json
from dotenv import load_dotenv
from google import genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Initialize ChromaDB Vector Store
DB_DIR = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

TEST_DATASET = [
    {
        "question": "How does user authentication work and what is the test password?",
        "expected_keyword": "secret123"
    },
    {
        "question": "How do you fetch a user profile from the database?",
        "expected_keyword": "fetch_user_profile"
    }
]

def evaluate_rag_pipeline():
    print("🧪 Running Automated RAG Evaluation Suite...\n" + "="*50)
    
    total_tests = len(TEST_DATASET)
    retrieval_passes = 0
    faithfulness_scores = []

    for idx, item in enumerate(TEST_DATASET, 1):
        question = item["question"]
        expected_keyword = item["expected_keyword"]

        print(f"\n[Test {idx}/{total_tests}] Question: '{question}'")

        # 1. Retrieve Context from ChromaDB
        retrieved_docs = vector_db.similarity_search(question, k=2)
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

        # Evaluate Retrieval Precision by checking if the expected function/keyword was retrieved
        retrieval_success = expected_keyword in retrieved_text
        if retrieval_success:
            retrieval_passes += 1
            print("  ✅ Retrieval Test: PASSED (Target code snippet retrieved)")
        else:
            print("  ❌ Retrieval Test: FAILED (Target keyword not found in context)")

        # 2. Generate Answer using Gemini
        rag_prompt = f"""
Answer the question using ONLY the code context provided.
Context:
{retrieved_text}

Question:
{question}
"""
        gen_response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=rag_prompt
        )
        ai_answer = gen_response.text.strip()

        # 3. Evaluate Faithfulness using LLM-as-a-Judge
        eval_prompt = f"""
You are an unbiased AI Evaluator. 
Score how FAITHFUL and ACCURATE the generated answer is to the retrieved code context on a scale of 0 to 100.

Retrieved Context:
{retrieved_text}

Generated Answer:
{ai_answer}

User Question:
{question}

Instructions:
- Return ONLY a valid JSON object in this exact format: {{"faithfulness_score": 100, "reason": "short explanation"}}
- Score 100 if the answer strictly relies on the context without inventing facts.
- Score 0 if it contains hallucinations or contradicts the code context.
"""
        try:
            eval_response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=eval_prompt
            )
            raw_text = eval_response.text.strip().replace("```json", "").replace("```", "")
            eval_data = json.loads(raw_text)
            score = eval_data.get("faithfulness_score", 0)
            reason = eval_data.get("reason", "No reason provided")
            
            faithfulness_scores.append(score)
            print(f"  🧠 Faithfulness Score: {score}/100")
            print(f"  📝 Judge Note: {reason}")
        except Exception as e:
            print(f"  ⚠️ Evaluation Judge Error: {e}")

    # Summary Report
    retrieval_accuracy = (retrieval_passes / total_tests) * 100
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0

    print("\n" + "="*50)
    print("📊 FINAL RAG EVALUATION REPORT")
    print("="*50)
    print(f"• Context Retrieval Precision: {retrieval_accuracy:.1f}%")
    print(f"• Average LLM Faithfulness Score: {avg_faithfulness:.1f}/100")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_rag_pipeline()