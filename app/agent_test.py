from app.agent import ask_agent


question = "Quelles sont les sanctions disciplinaires ?"


result = ask_agent(question)


print("\nRÉPONSE")
print("=" * 50)
print(result["answer"])


print("\nSOURCES")
print("=" * 50)

for source in result["sources"]:
    print(
        "Page:",
        source["page"],
        "| Source:",
        source["source"]
    )