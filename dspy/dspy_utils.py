import dspy
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'),temperature=0.5, cache=False)
dspy.configure(lm=lm)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

class PersonalizedSummary(dspy.Signature):
    document: str = dspy.InputField(desc="The document to summarize")
    length: int = dspy.InputField(desc="The maximum length of the summary")
    style: str = dspy.InputField(default="formal", desc="The style of the summary, like 'formal', 'casual'")
    context: str = dspy.InputField(default=False, desc="May contain relevant facts")
    summary: str = dspy.OutputField(desc="The summary of the document")

class SummaryPipeline(dspy.Module):
    def __init__(self, wiki_knowledge=False):
        self.wiki_knowledge = wiki_knowledge
        if wiki_knowledge:
            self.retriever = dspy.Retrieve(k=3)
        self.summarizer = dspy.ChainOfThought(PersonalizedSummary)
    
    def forward(self, document, content_preference="", length=50, style="formal"):
        wiki_knowledge=""
        if self.wiki_knowledge:
            # Retrieve relevant documents
            wiki_knowledge = self.retriever(query=document).passages
        # Summarize the relevant documents
        summary = self.summarizer(document=document, content_preference=content_preference, length=length, style=style, context=wiki_knowledge).summary
        return summary

class Assess(dspy.Signature):
    original_document: str = dspy.InputField(desc="The original document to summarize")
    summary = dspy.InputField(desc="The summary text, evaluated based on whether the length, and style personalization preferences are met")
    length: int = dspy.InputField(desc="The maximum length of the summary")
    style: str = dspy.InputField(default="formal", desc="The style of the summary, either 'formal' or 'casual'")
    score: int = dspy.OutputField(desc="The score of the summary, between 0 and 10")

class GeneralAssess(dspy.Signature):
    original_document: str = dspy.InputField(desc="The original document to summarize")
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

def metric(gt, summary,trace=None):
    document, gt_summary, length,style = gt.document, gt.highlights, gt.length, gt.style

    # assessment_question
    assessment_question1 = "Is the summary of the document be inclusive of the main points?"
    assessment_question2 = f"Is the summary of the document around `{length}` words?"
    assessment_question3 = f"Is the summary of the document in a `{style}` style?"
    assessment_question4 = f"Compared to the ground truth summary `{gt_summary}`, is the summary of the document as good as the ground truth summary?"

    q1 = dspy.Predict(GeneralAssess)(original_document=document, assessed_text=summary, assessment_question=assessment_question1)
    q2 = dspy.Predict(GeneralAssess)(original_document=document, assessed_text=summary, assessment_question=assessment_question2)
    q3 = dspy.Predict(GeneralAssess)(original_document=document, assessed_text=summary, assessment_question=assessment_question3)
    q4 = dspy.Predict(GeneralAssess)(original_document=document, assessed_text=summary, assessment_question=assessment_question4)

    q1, q2, q3, q4=(m.assessment_answer.split()[0].lower() == 'yes' for m in [q1, q2, q3, q4])
    score= (q1+q2+q3+q4)/4*10
    return score

def retrieve_wiki_knowledge(query):
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

def summarize_df(df,length=50,style="formal"):
    # empty dictionary to store the results
    results = {}
    documents = []
    summaries = []
    summary_scores = []
    gt_scores = []

    summarize = dspy.ChainOfThought(PersonalizedSummary)
    assess = dspy.ChainOfThought(Assess)
    # for each row, get the document and the summary
    for i in range(len(df)):
        document = df.iloc[i]['article']
        summary = summarize(document=document,length=length,style=style,context="").summary

        gt_summary = df.iloc[i]['highlights']
        summary_score = assess(original_document=document,summary=summary, length=length, style=style, context="").score
        gt_score = assess(original_document=document,summary=gt_summary, length=length, style=style, context="").score

        documents.append(document)
        summaries.append(summary)
        summary_scores.append(summary_score)
        gt_scores.append(gt_score)
    
    results['documents'] = documents
    results['summaries'] = summaries
    results['summary_scores'] = summary_scores
    results['gt_scores'] = gt_scores
    return results
