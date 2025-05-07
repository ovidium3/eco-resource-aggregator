from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv

from kg_client import top_three
from rewrite_pipeline import doPipeline
import json, os
from pathlib import Path

CLIMATE_DIR = Path(__file__).resolve().parent.parent / "climate_outputs"

def get_paper_text_and_title(category: str, paper_id: int | str):
    """
    Return (fullText, title) for the paper with `paper_id`
    inside <climate_outputs>/<category>.json.
    Raises FileNotFoundError or ValueError if not found.
    """
    fp = CLIMATE_DIR / f"{category}.json"
    if not fp.exists():
        raise FileNotFoundError(f"No file {fp}")

    with fp.open("r", encoding="utf‑8") as f:
        papers = json.load(f)

    for item in papers:
        if str(item.get("id")) == str(paper_id):
            return item.get("fullText", ""), item.get("title", "Unknown title")

    raise ValueError(f"id {paper_id} not found in {fp}")

load_dotenv()


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY") 

# Note - this is the only endpoint (besides the '/') that is used. The rest is debugging
# This accepts the query from the ui, then rewrites it to better fit our system
# Then we categorize the query using pytorch transformers
# Next we call the neo4j in order to get the results
# We use the id in order to get some important data and include that in our summary
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({ "reply": "Empty message." })
    '''
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # default for chat completion
            messages=[
                { "role": "user", "content": user_msg }
            ],
            max_tokens = 200 # unsure if needed
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Server error: {e}"

    return jsonify({ "reply": reply })
    '''
    
    cat_print, rewritten, query = doPipeline(user_msg)

    reply = f"{cat_print} {rewritten} {query}"
    print(reply)
    
    retrieved = top_three(cat_print, rewritten)
    finalResponse = f"Sources: \n \n 1. {retrieved[0]["title"]} \n"
    print(retrieved)


    try:
        full_text1, paper_title1 = get_paper_text_and_title(cat_print, retrieved[0]['id'])
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    
    try:
        full_text2, paper_title2 = get_paper_text_and_title(cat_print, retrieved[1]['id'])
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    try:
        full_text3, paper_title3 = get_paper_text_and_title(cat_print, retrieved[2]['id'])
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


    try:
        response = openai.ChatCompletion.create(
            model="o1", #Change this depending on what we're feeling
            messages=[
                { "role": "user", "content": f"Briefly answer this question in one paragraph - {user_msg} - based on these three documents: [title]: {paper_title1} [full text]: {full_text1} \n \n [title]: {paper_title2} [full text]: {full_text2} \n \n [title]: {paper_title3} [full text]: {full_text3}. Begin your response with 'Based on the three most relevant documents in our database...' Include inline citations using the three titles that I provided to you. Try to add the year and the authors if you can"}
            ],
            max_completion_tokens = 20000 # Unsure if needed
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Server error: {e}"
    finalReply = f"We've sorted your query into the '{cat_print}' category. The fully modified query is: '{rewritten.partition("> ")[2].strip()}'. \n Find our answer here: {reply} Also, here is the full return (with scores) for transparency: {str(retrieved)}"
    return jsonify({ "reply": finalReply })

@app.get("/papers")
def papers():
    cat  = request.args.get("category")
    text = request.args.get("query", "")
    if not cat:
        return {"error": "category param missing"}, 400
    return jsonify(top_three(cat, text))

@app.get("/queryPros")
def queryPros():
    query  = request.args.get("query")
    if not query:
        return {"error": "category param missing"}, 400
    return jsonify(doPipeline(query))

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
