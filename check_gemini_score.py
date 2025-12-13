
import os
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import getpass

def check_gemini_score():
    print("This script checks the similarity score using Google Gemini.")
    api_key = input("Please enter your Google API Key: ").strip()
    
    if not api_key:
        print("API Key required.")
        return

    genai.configure(api_key=api_key)
    
    # User's query
    query = "छीन लेना"
    
    # Target question
    target_question = "राधेश्याम बाबाजी दंडवत प्रणाम... एक प्रश्न था.. एक सिद्धांत ये बोलता है कि भगवान के भक्तों से भगवान उनका सब कुछ ले लेते हैं.. और बाबाजी के सत्संग में बाबा ने बताया भगवान के भक्तों को भगवान धन धान्य से भर देते हैं . दोनों सिद्धांत समझ में नहीं आए, कृपा थोड़ी स्पष्ट कीजिए🙏🏻"
    
    print("\nCalculating embeddings...")
    try:
        # Embedding the query
        # task_type="retrieval_query"
        q_resp = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        q_emb = np.array([q_resp['embedding']])
        
        # Embedding the document
        # task_type="retrieval_document"
        d_resp = genai.embed_content(
            model="models/text-embedding-004",
            content=target_question,
            task_type="retrieval_document"
        )
        d_emb = np.array([d_resp['embedding']])
        
        score = cosine_similarity(q_emb, d_emb)[0][0]
        
        print(f"\nQuery: {query}")
        print(f"Target: {target_question[:50]}...")
        print(f"Gemini Similarity Score: {score}")
        print(f"Current App Threshold for Gemini: 0.50")
        
        if score < 0.50:
            print("FAIL: Score is below 0.50, so it is hidden.")
            print("RECOMMENDATION: Lower the threshold in app.py")
        else:
            print("SUCCESS: Score is above 0.50. It should have appeared if top_k limit wasn't reached.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_gemini_score()
