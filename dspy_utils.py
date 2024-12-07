import dspy
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'),temperature=0.7)
dspy.configure(lm=lm)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

class PersonalizedSummary(dspy.Signature):
    document: str = dspy.InputField(desc="The document to summarize")
    length: int = dspy.InputField(desc="The maximum length of the summary")
    style: str = dspy.InputField(default="formal", desc="The style of the summary, like 'formal', 'casual'")
    context: str = dspy.InputField(default=False, desc="May contain relevant facts")
    summary: str = dspy.OutputField(desc="The summary of the document")

class KeyWords(dspy.Signature):
    document: str = dspy.InputField(desc="The document to extract keywords from")
    keywords: list = dspy.OutputField(desc="The top 5 keywords extracted from the document")

class SummaryPipeline(dspy.Module):
    def __init__(self, wiki_knowledge=False):
        self.wiki_knowledge = wiki_knowledge
        if wiki_knowledge:
            self.retriever = dspy.Retrieve(k=3)
            self.keywords = dspy.ChainOfThought(KeyWords)
        self.summarizer = dspy.ChainOfThought(PersonalizedSummary)
    
    def forward(self, document, content_preference="", length=50, style="formal"):
        wiki_knowledge=""
        if self.wiki_knowledge:
            key_words = self.keywords(document=document).keywords
            # Retrieve relevant documents
            wiki_knowledge = self.retriever(query=key_words).passages
        # Summarize the relevant documents
        summary = self.summarizer(document=document, content_preference=content_preference, length=length, style=style, context=wiki_knowledge).summary
        return summary

class Assess(dspy.Signature):
    original_document: str = dspy.InputField(desc="The original document to summarize")
    summary = dspy.InputField(desc="The summary text, evaluated based on whether the length, and style personalization preferences are met")
    length: int = dspy.InputField(desc="The maximum length of the summary")
    style: str = dspy.InputField(default="formal", desc="The style of the summary, either 'formal' or 'casual'")
    score: int = dspy.OutputField(desc="The score of the summary, between 0 and 100")
