"""
Intelligent Candidate Discovery Platform - Hackathon MVP
Semantic matching of Job Descriptions to Candidate Resumes using BERT embeddings
"""

import streamlit as st
import PyPDF2
import fitz  # PyMuPDF for better PDF parsing
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple
import os

# Page config to set browser page title
st.set_page_config(page_title="SmartHire AI", layout="wide")

@st.cache_resource
def load_model():
    """Load pre-trained SentenceTransformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF using PyMuPDF (better than PyPDF2)"""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except:
        return ""

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s\.\,\-\(\)]', '', text)  # Remove special chars
    return text.strip().lower()

def extract_skills(text: str) -> List[str]:
    """Extract key skills using simple regex patterns"""
    skill_patterns = [
        r'(python|java|javascript|react|angular|node\.?js?|aws|azure|gcp|docker|kubernetes)',
        r'(sql|mysql|postgresql|mongodb|redis)',
        r'(machine learning|deep learning|nlp|computer vision)',
        r'(devops|ci\/cd|jenkins|gitlab|github actions)'
    ]
    skills = []
    for pattern in skill_patterns:
        skills.extend(re.findall(pattern, text))
    return list(set(skills))

def parse_job_description(text: str) -> Dict:
    """Parse JD into structured features"""
    cleaned = clean_text(text)
    skills = extract_skills(cleaned)
    
    # Experience estimation (heuristic)
    exp_keywords = ['years?', 'experience', 'yr']
    exp_score = sum(1 for kw in exp_keywords if kw in cleaned)
    
    return {
        'raw_text': cleaned,
        'skills': skills,
        'exp_score': exp_score,
        'text_length': len(cleaned)
    }

def parse_resume(text: str) -> Dict:
    """Parse resume into structured features"""
    cleaned = clean_text(text)
    skills = extract_skills(cleaned)
    
    # Experience indicators
    exp_indicators = ['months?', 'years?', 'experience']
    exp_score = sum(1 for ind in exp_indicators if ind in cleaned)
    
    return {
        'raw_text': cleaned,
        'skills': skills,
        'exp_score': exp_score,
        'text_length': len(cleaned)
    }

def compute_match_score(jd_features: Dict, resume_features: Dict, 
                       embedding_similarity: float) -> Dict:
    """Compute hybrid match score: semantic + skills + experience"""
    
    # Skill overlap score
    jd_skills = set(jd_features['skills'])
    resume_skills = set(resume_features['skills'])
    skill_overlap = len(jd_skills.intersection(resume_skills)) / len(jd_skills) if jd_skills else 0
    
    # Experience alignment (normalized)
    exp_align = 1 - abs(jd_features['exp_score'] - resume_features['exp_score']) / max(jd_features['exp_score'], 1)
    
    # Weighted hybrid score
    semantic_weight = 0.6
    skill_weight = 0.25
    exp_weight = 0.15
    
    hybrid_score = (
        semantic_weight * embedding_similarity +
        skill_weight * skill_overlap +
        exp_weight * exp_align
    )
    
    return {
        'semantic_score': embedding_similarity,
        'skill_overlap': skill_overlap,
        'exp_alignment': exp_align,
        'hybrid_score': hybrid_score
    }

def main():
    st.title("🚀 SmartHire AI - Intelligent Candidate Matching")
    st.markdown("**Semantic matching of Job Descriptions to Candidate Resumes**")
    
    model = load_model()
    
    # Sidebar for file uploads
    st.sidebar.header("Upload Documents")
    
    # Job Description Upload
    jd_file = st.sidebar.file_uploader("📄 Job Description (PDF)", 
                                     type=['pdf'], key="jd")
    
    # Candidate Resumes (multiple)
    resume_files = st.sidebar.file_uploader("📋 Candidate Resumes (PDFs)", 
                                          type=['pdf'], 
                                          accept_multiple_files=True,
                                          key="resumes")
    
    if jd_file and len(resume_files) > 0:
        with st.spinner("🔄 Processing documents..."):
            # Parse Job Description
            jd_text = extract_text_from_pdf(jd_file)
            if not jd_text:
                st.error("❌ Could not extract text from Job Description")
                return
            
            jd_features = parse_job_description(jd_text)
            st.success(f"✅ JD processed: {len(jd_features['skills'])} skills detected")
            
            # Process all resumes
            candidates = []
            jd_embedding = model.encode(jd_features['raw_text'])
            
            for resume_file in resume_files:
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    resume_features = parse_resume(resume_text)
                    
                    # Semantic similarity
                    resume_embedding = model.encode(resume_features['raw_text'])
                    semantic_sim = util.cos_sim(jd_embedding, resume_embedding)[0][0].item()
                    
                    # Hybrid scoring
                    match_scores = compute_match_score(jd_features, resume_features, semantic_sim)
                    
                    candidates.append({
                        'filename': resume_file.name,
                        'features': resume_features,
                        'scores': match_scores,
                        'top_skills_match': list(set(jd_features['skills']) & set(resume_features['skills']))[:5]
                    })
            
            # Rank candidates
            candidates.sort(key=lambda x: x['scores']['hybrid_score'], reverse=True)
            
            # Display Results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("🏆 Top Candidate Matches")
                df_data = []
                for i, cand in enumerate(candidates[:10]):
                    df_data.append({
                        'Rank': i+1,
                        'Candidate': cand['filename'][:30] + "...",
                        'Match Score': f"{cand['scores']['hybrid_score']:.1%}",
                        'Semantic': f"{cand['scores']['semantic_score']:.1%}",
                        'Skills': f"{cand['scores']['skill_overlap']:.1%}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            
            with col2:
                st.header("📊 Match Breakdown")
                if candidates:
                    top_cand = candidates[0]
                    st.metric("Best Match Score", f"{top_cand['scores']['hybrid_score']:.1%}")
                    st.metric("Top Skills Match", 
                            ", ".join(top_cand['top_skills_match']) or "None")
            
            # Detailed view for top 3
            st.header("🔍 Detailed Analysis (Top 3)")
            for i, candidate in enumerate(candidates[:3]):
                with st.expander(f"#{i+1} {candidate['filename']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{candidate['scores']['hybrid_score']:.1%}")
                    with col2:
                        st.metric("Semantic Similarity", f"{candidate['scores']['semantic_score']:.1%}")
                    with col3:
                        st.metric("Skill Overlap", f"{candidate['scores']['skill_overlap']:.1%}")
                    
                    st.write("**Matching Skills:**", 
                            ", ".join(candidate['top_skills_match']) or "No direct matches")
                    st.caption(f"Experience alignment: {candidate['scores']['exp_alignment']:.1%}")
    
    elif jd_file:
        st.warning("👆 Please upload candidate resumes to see matches")
    elif resume_files:
        st.warning("👆 Please upload a Job Description first")
    else:
        st.info("📤 Upload a Job Description and Candidate Resumes to get started!")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload Job Description**: PDF format
        2. **Upload Resumes**: Multiple PDFs supported  
        3. **Get Instant Matches**: Semantic + skills-based ranking
        4. **Review Details**: Breakdown of why candidates match
        
        **Features:**
        - BERT-powered semantic understanding
        - Skill extraction & overlap detection
        - Experience alignment scoring
        - Bias-reduced matching (skills > keywords)
        """)

if __name__ == "__main__":
    main()
