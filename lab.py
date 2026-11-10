from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


# Part A: mock API example (no network request is made)
mock_request = {
    "model": "example-chat-model",
    "messages": [
        {"role": "system", "content": "Answer clearly and use only supplied context."},
        {"role": "user", "content": "What is RAG?"}
    ],
    "temperature": 0.2
}
mock_response = {
    "id": "demo-response-001",
    "output_text": "RAG retrieves relevant information and supplies it to a model as context.",
    "usage": {"input_tokens": 24, "output_tokens": 16}
}
print("MOCK API REQUEST/RESPONSE")
print(json.dumps({"request": mock_request, "response": mock_response}, indent=2))

# Part B: local sample knowledge base
documents = [
    {"source": "cloud_basics.txt",
     "text": "Amazon EC2 provides resizable compute capacity in the AWS cloud. "
             "Amazon S3 is object storage used to store and retrieve data. "
             "IAM controls access to AWS resources using identities and policies."},
    {"source": "rag_basics.txt",
     "text": "Retrieval-Augmented Generation, or RAG, retrieves relevant passages from a knowledge source "
             "and adds them to the prompt as context. Chunking divides long documents into smaller pieces. "
             "Retrieval quality depends on source quality, chunk design, and query relevance."},
    {"source": "security_basics.txt",
     "text": "Use least privilege when granting access. Do not place passwords, access tokens, or API keys "
             "in source code. Store secrets in a suitable secret manager or environment variable."}
]

def chunk_text(text, chunk_size=180, overlap=30):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks

chunks = []
for document in documents:
    for index, part in enumerate(chunk_text(document["text"])):
        chunks.append({
            "source": document["source"],
            "chunk_id": f'{document["source"]}#{index + 1}',
            "text": part
        })

# Part C: local TF-IDF retrieval
vectorizer = TfidfVectorizer(stop_words="english")
document_matrix = vectorizer.fit_transform([item["text"] for item in chunks])

def retrieve(query, top_k=2):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_matrix).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]
    return [
        {**chunks[idx], "score": float(scores[idx])}
        for idx in ranked_indices if scores[idx] > 0
    ]

def answer_from_retrieval(question, top_k=2):
    results = retrieve(question, top_k)
    if not results:
        return {"answer": "I could not find relevant information in the local knowledge base.",
                "sources": []}
    context = "\n".join(
        f'[{item["chunk_id"]}] {item["text"]}' for item in results
    )
    return {"answer": "Based on the retrieved context:\n" + context,
            "sources": [item["chunk_id"] for item in results]}

# Part D: simulated tool/function
def lookup_service(service_name):
    services = {
        "EC2": "Virtual compute capacity",
        "S3": "Object storage",
        "IAM": "Identity and access management"
    }
    return services.get(service_name.strip().upper(),
                        "Unknown service. Check the service name.")

print("\nLOCAL RAG DEMO")
question = input("Ask a question about the sample documents: ").strip()
result = answer_from_retrieval(question)
print("\nAnswer/context:\n", result["answer"])
print("Sources:", ", ".join(result["sources"]) if result["sources"] else "None")

print("\nSIMULATED TOOL CALL")
tool_call = {"name": "lookup_service", "arguments": {"service_name": "S3"}}
if tool_call["name"] == "lookup_service":
    print(lookup_service(**tool_call["arguments"]))