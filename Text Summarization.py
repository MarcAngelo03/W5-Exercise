from transformers import pipeline

# Initialize text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Text to summarize
text = """
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
especially computer systems. These processes include learning, reasoning, and self-correction.
AI is widely used in healthcare, finance, education, and business industries.
"""

# Prompt
prompt = f"""
Summarize the following text in one sentence:

{text}

Summary:
"""

# Generate summary
output = generator(
    prompt,
    max_length=80,
    temperature=0.7,
    top_p=0.95,
    num_return_sequences=1
)

#print("=== SUMMARY ===")
print(output[0]["generated_text"])
