"""

EU AI Act Assistant — a RAG chatbot using a Gradio user interface.

Features:
- Welcome screen with tips and popular topics
- Can answer questions based on the EU AI Act document
- LLM-as-judge used to verify answer quality
- Confidence score based on vector similarity
- Suggested follow-up questions
- "I don't know" handling when context is weak

"""

import os
import sys
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import AzureOpenAI
from dotenv import load_dotenv
import gradio as gr

# Allow imports from project root when running from jose_dev
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Load environment variables ─────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
DEPLOY_LARGE = os.environ["DEPLOY_LARGE"]
DEPLOY_GPT = os.environ["DEPLOY_GPT"]

PGUSER = os.environ["PGUSER"]
PGPASSWORD = os.environ["PGPASSWORD"]
PGHOST = os.environ["PGHOST"]
PGPORT = os.getenv("PGPORT", "5433")
PGDATABASE = os.environ["PGDATABASE"]

TOP_K = 5

# Similarity threshold — below this we warn the user context may be weak
CONFIDENCE_THRESHOLD = 0.75

# ─── Popular topics shown when user types 'tips' ────
POPULAR_TOPICS = [
    "🔴 1. Prohibited AI practices — what is completely banned?",
    "⚠️ 2.  High-risk AI systems — which systems are considered high risk?",
    "🏛️ 3. Governance — who enforces the EU AI Act?",
    "🔍 4. Transparency — what must AI providers disclose?",
    "🤖 5. General purpose AI — how are large AI models regulated?",
]

WELCOME_MESSAGE = """**Hi! I'm an assistant specialised in the EU AI Act.**

I can help you understand the regulation, from prohibited practices to compliance obligations.

**Commands:**
- Type 'tips' for tips on popular topics you can ask about
- Type 'exit' to end the session
- Or just ask me anything about the EU AI Act!

**Example questions:**
- What AI practices are completely banned?
- How will the AI Act affect companies outside the EU?
- What are the rules for high-risk AI systems?
"""


# ─── Set up clients ─────
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

def get_connection():
    return psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        user=PGUSER,
        password=PGPASSWORD,
        dbname=PGDATABASE
    )


# ─── Embed a question ────
def embed_question(question):
    response = client.embeddings.create(
        input=question,
        model=DEPLOY_LARGE
    )
    return response.data[0].embedding


# ─── Search for relevant chunks ────
def search_similar_chunks(question_embedding, top_k=TOP_K):
    conn = get_connection()
    with conn.cursor() as cur:
        register_vector(cur)
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM knowledge_base
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (question_embedding, question_embedding, top_k))
        results = cur.fetchall()
    conn.close()
    return results  # list of (content, similarity_score)


# ─── Generate answer with GPT-4 ──────
def generate_answer(question, chunks):
    context = "\n\n---\n\n".join([chunk for chunk, score in chunks])

    system_prompt = """You are a factual assistant specialised in the EU AI Act.
Answer questions based ONLY on the provided context from the document.
Be precise and concise. 
If the context does not contain sufficient information, say clearly:
'I could not find enough information in the EU AI Act to answer this confidently.'
Never make up or infer information not present in the context."""

    user_prompt = f"""Context from the EU AI Act:

{context}

---

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=DEPLOY_GPT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return response.choices[0].message.content


# ─── LLM-as-judge: verify the answer is grounded ──────
def judge_answer(question, answer, chunks, top_score):
    """
    Three-part judge:
    1. Are the retrieved chunks actually relevant to the question?
    2. Is the answer supported by those chunks?
    3. Is the confidence score high enough to trust the answer?
    """
    # Fail immediately if similarity score is too low
    if top_score < 0.50:
        return False, "retrieval", "The retrieved context does not appear relevant to this question."

    context = "\n\n---\n\n".join([chunk for chunk, score in chunks])

    judge_prompt = f"""You are a strict quality judge for a RAG system built on the EU AI Act.

You must evaluate TWO things:

1. RELEVANCE: Are the context chunks actually relevant to the question?
2. GROUNDED: Is every claim in the answer directly supported by the context?

Context chunks:
{context}

Question: {question}

Answer: {answer}

Respond in this exact format:
RELEVANCE: yes or no
GROUNDED: yes or no
ISSUES: describe any problems, or write 'none'"""

    response = client.chat.completions.create(
        model=DEPLOY_GPT,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_tokens=200
    )

    result = response.choices[0].message.content.strip().lower()
    is_relevant = "relevance: yes" in result
    is_grounded = "grounded: yes" in result
    issues_line = [l for l in result.split("\n") if l.startswith("issues:")]
    issues = issues_line[0].replace("issues:", "").strip() if issues_line else "none"

    if not is_relevant:
        return False, "relevance", "The document sections found don't appear relevant to your question."
    if not is_grounded:
        return False, "grounded", f"The answer may contain unsupported claims: {issues}"

    return True, "ok", "none"


# ─── Suggest follow-up questions ─────
def suggest_followups(question, answer):
    response = client.chat.completions.create(
        model=DEPLOY_GPT,
        messages=[{
            "role": "user",
            "content": f"""Based on this question and answer about the EU AI Act, suggest 3 short follow-up questions the user might want to ask next.
Return ONLY the 3 questions as a numbered list, nothing else.

Question: {question}
Answer: {answer}"""
        }],
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


# ─── Confidence label from similarity score ──────
def confidence_label(score):
    if score >= 0.90:
        return "🟢 Very high confidence"
    elif score >= 0.80:
        return "🟢 High confidence"
    elif score >= 0.70:
        return "🟡 Moderate confidence"
    elif score >= 0.60:
        return "🟠 Low confidence — answer may be incomplete"
    else:
        return "🔴 Very low confidence — limited relevant content found"


# ─── Main chat function ──────
def chat(user_message, history):
    user_message = user_message.strip()

    # Handle special commands
    if user_message.lower() == "exit":
        return history + [[user_message, "👋 Bye! Thanks for using the EU AI Act assistant."]]

    if user_message.lower() == "tips":
        topics_text = "**Popular topics you can ask about:**\n\n" + "\n".join(POPULAR_TOPICS)
        return history + [[user_message, topics_text]]

    if not user_message:
        return history

    # Normal question flow
    try:
        # Embed the question
        question_embedding = embed_question(user_message)

        # Find relevant chunks
        chunks = search_similar_chunks(question_embedding)
        top_score = chunks[0][1] if chunks else 0

        # Generate answer
        answer = generate_answer(user_message, chunks)

        # Judge the answer
        is_grounded, judge_type, issues = judge_answer(user_message, answer, chunks, top_score)

        # Suggest follow-ups
        followups = suggest_followups(user_message, answer)

        # Build the full response
        confidence = confidence_label(top_score)

        response_parts = [answer]

        # Add grounding warning if needed
        if not is_grounded:
            if judge_type == "retrieval":
                response_parts.append(
                    f"\n\n🔴 **Low confidence warning:** I couldn't find strongly relevant sections in the EU AI Act for this question. The answer above may not be reliable — please verify against the original document."
                )
            elif judge_type == "relevance":
                response_parts.append(
                    f"\n\n⚠️ **Relevance warning:** The document sections I found may not directly address your question. Consider rephrasing."
                )
            elif judge_type == "grounded":
                response_parts.append(
                    f"\n\n⚠️ **Quality note:** The answer may contain claims not directly supported by the EU AI Act: {issues}"
                )
        # Add confidence score
        response_parts.append(f"\n\n**Confidence:** {confidence} ({top_score:.0%})")

        # Add follow-up suggestions
        response_parts.append(f"\n\n**You may also be interested in:**\n{followups}")

        full_response = "".join(response_parts)

        return history + [[user_message, full_response]]

    except Exception as e:
        error_msg = f"Something went wrong: {str(e)}"
        return history + [[user_message, error_msg]]


# ─── 1Gradio UI ───────
with gr.Blocks(theme=gr.themes.Soft(), title="EU AI Act Assistant") as demo:

    gr.Markdown("""
    # 🇪🇺 EU AI Act Assistant
    *Powered by RAG — answers grounded in the official EU AI Act document*
    """)

    chatbot = gr.Chatbot(
        value=[[None, WELCOME_MESSAGE]],
        height=550,
        label="EU AI Act Assistant",
        bubble_full_width=False
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about the EU AI Act, or type 'tips' for popular topics...",
            scale=9,
            show_label=False,
            container=False
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    with gr.Row():
        clear_btn = gr.Button("Clear conversation", size="sm")

    # Handle send
    submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(lambda: "", outputs=msg)

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(lambda: "", outputs=msg)

    clear_btn.click(
        fn=lambda: [[None, WELCOME_MESSAGE]],
        outputs=chatbot
    )

if __name__ == "__main__":
    print("Starting EU AI Act Assistant...")
    print("Open your browser at http://localhost:7867\n")
    demo.launch(server_name="0.0.0.0", server_port=7867)