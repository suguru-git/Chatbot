# ========================================
# COMPLETE PDF/YOUTUBE TRANSCRIPT CHATBOT
# ========================================
# This is your MAIN file - save as 'chatbot.py'

import os
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import PyPDF2
import json
from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs

# ========================================
# GLOBAL SETUP - Initialize once
# ========================================
print("🔄 Loading AI model... (This takes 30 seconds first time)")
model = SentenceTransformer("all-MiniLM-L6-v2")
DB_PATH = "knowledge_base.pkl"
CONVERSATION_PATH = "conversation_history.pkl"

# ========================================
# PART 1: FILE READING FUNCTIONS
# ========================================
# This section handles PDF and TXT file reading

def read_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"❌ Error reading PDF: {str(e)}")
    return text

def read_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        raise Exception(f"❌ Error reading TXT: {str(e)}")
    return text

def read_file(file_path):
    """Main function to read any supported file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    if file_size > 1024:  # 1GB limit
        raise ValueError(f"❌ File too large: {file_size:.1f}MB. Max 1GB allowed.")
    
    print(f"📄 Reading file: {os.path.basename(file_path)} ({file_size:.1f}MB)")
    
    if file_path.lower().endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        return read_txt(file_path)
    else:
        raise ValueError("❌ Unsupported file. Use PDF or TXT files only.")

# ========================================
# PART 2: YOUTUBE TRANSCRIPT FUNCTIONS
# ========================================
# This section handles YouTube video transcript extraction

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(url):
    """Get transcript from YouTube video"""
    print(f"🎥 Fetching transcript from: {url}")
    
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("❌ Invalid YouTube URL")
    
    try:
        # Try to get transcript in English first
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        # Combine all transcript parts
        full_text = ""
        for entry in transcript_list:
            full_text += entry['text'] + " "
        
        print(f"✅ Transcript extracted: {len(full_text)} characters")
        return full_text
        
    except Exception as e:
        # Try auto-generated transcript if manual not available
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en-GB'])
            full_text = ""
            for entry in transcript_list:
                full_text += entry['text'] + " "
            print(f"✅ Auto-generated transcript extracted: {len(full_text)} characters")
            return full_text
        except:
            raise Exception(f"❌ No transcript available for this video: {str(e)}")

# ========================================
# PART 3: TEXT PROCESSING FUNCTIONS
# ========================================
# This section handles text chunking and embedding

def chunk_text(text, max_words=150):
    """Split text into smaller chunks for better processing"""
    print("🔄 Chunking text...")
    
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into words
    words = text.split()
    
    # Create overlapping chunks for better context
    chunks = []
    overlap = 30  # 30 words overlap between chunks
    
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 20:  # Only add non-empty chunks
            chunks.append(chunk)
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def embed_chunks(chunks):
    """Convert text chunks to embeddings"""
    print("🔄 Creating embeddings...")
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    print(f"✅ Embeddings created: {embeddings.shape}")
    return embeddings

# ========================================
# PART 4: DATABASE FUNCTIONS
# ========================================
# This section handles saving and loading data

def save_to_database(chunks, embeddings, source_info):
    """Save chunks and embeddings to database"""
    print("💾 Saving to database...")
    
    data = {
        'chunks': chunks,
        'embeddings': embeddings,
        'source_info': source_info,
        'created_at': str(np.datetime64('now'))
    }
    
    with open(DB_PATH, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Database saved with {len(chunks)} chunks")

def load_database():
    """Load database if exists"""
    if not os.path.exists(DB_PATH):
        return None
    
    try:
        with open(DB_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"📚 Database loaded: {len(data['chunks'])} chunks from {data['source_info']}")
        return data
    except Exception as e:
        print(f"❌ Error loading database: {str(e)}")
        return None

def save_conversation(conversation):
    """Save conversation history"""
    with open(CONVERSATION_PATH, 'wb') as f:
        pickle.dump(conversation, f)

def load_conversation():
    """Load conversation history"""
    if os.path.exists(CONVERSATION_PATH):
        try:
            with open(CONVERSATION_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            return []
    return []

# ========================================
# PART 5: AI QUERY FUNCTIONS
# ========================================
# This section handles finding relevant content and generating answers

def find_best_context(question, knowledge_base, top_k=3):
    """Find most relevant chunks for the question"""
    print(f"🔍 Searching for: {question[:50]}...")
    
    # Embed the question
    question_embedding = model.encode([question], convert_to_tensor=True)
    
    # Calculate similarities
    similarities = cosine_similarity(question_embedding, knowledge_base['embeddings'])[0]
    
    # Get top K most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    contexts = []
    for idx in top_indices:
        contexts.append({
            'text': knowledge_base['chunks'][idx],
            'score': similarities[idx]
        })
    
    print(f"✅ Found {len(contexts)} relevant contexts (best score: {contexts[0]['score']:.3f})")
    return contexts

def ask_ollama(prompt, model_name="phi"):
    """Send prompt to Ollama for response"""
    try:
        print("🤖 Generating response...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', "❌ No response generated")
        else:
            return f"❌ Ollama error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Make sure it's running: ollama run phi"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def build_prompt(contexts, question, conversation_history):
    """Build comprehensive prompt with context and history"""
    
    # Add conversation history (last 3 exchanges)
    history_text = ""
    if conversation_history:
        history_text = "\n--- Previous Conversation ---\n"
        for q, a in conversation_history[-3:]:
            history_text += f"User: {q}\nAI: {a}\n\n"
    
    # Combine contexts
    context_text = ""
    for i, ctx in enumerate(contexts):
        context_text += f"Context {i+1} (relevance: {ctx['score']:.3f}):\n{ctx['text']}\n\n"
    
    prompt = f"""You are a helpful AI assistant. Use the provided context to answer the question accurately.

{history_text}

--- Relevant Context ---
{context_text}

--- Instructions ---
- Answer based on the context provided
- If context doesn't contain the answer, say "I don't have enough information"
- Keep responses clear and concise
- Maintain conversation flow

Question: {question}

Answer:"""
    
    return prompt

# ========================================
# PART 6: MAIN CHAT FUNCTIONS
# ========================================
# This section handles the main chatbot interaction

def chatbot_loop():
    """Main chatbot conversation loop"""
    db = load_database()
    if not db:
        print("❌ No knowledge base found. Please upload a document or YouTube video first.")
        return
    
    conversation = load_conversation()
    print(f"\n📚 Ready to chat about: {db['source_info']}")
    print("💬 Ask me anything! Type 'exit' to quit, 'clear' to clear history\n")
    
    while True:
        try:
            user_question = input("🧑 You: ").strip()
            
            if user_question.lower() in ['exit', 'quit', 'bye']:
                save_conversation(conversation)
                print("👋 Thanks for chatting! Goodbye!")
                break
            
            if user_question.lower() == 'clear':
                conversation = []
                save_conversation(conversation)
                print("🗑️ Conversation history cleared!")
                continue
            
            if not user_question:
                continue
            
            # Find relevant context
            contexts = find_best_context(user_question, db)
            
            # Build prompt with context and history
            prompt = build_prompt(contexts, user_question, conversation)
            
            # Get AI response
            ai_response = ask_ollama(prompt)
            
            # Display response
            print(f"\n🤖 AI: {ai_response}")
            print(f"📊 Best match score: {contexts[0]['score']:.3f}")
            print("-" * 50)
            
            # Save to conversation history
            conversation.append((user_question, ai_response))
            save_conversation(conversation)
            
        except KeyboardInterrupt:
            save_conversation(conversation)
            print("\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# ========================================
# PART 7: UPLOAD FUNCTIONS
# ========================================
# This section handles uploading new content

def upload_document():
    """Upload and process PDF/TXT document"""
    file_path = input("📄 Enter path to PDF or TXT file: ").strip().strip('"')
    
    try:
        # Read file
        text = read_file(file_path)
        
        if len(text.strip()) < 100:
            print("❌ File seems too short or empty")
            return
        
        # Process text
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        
        # Save to database
        source_info = f"File: {os.path.basename(file_path)}"
        save_to_database(chunks, embeddings, source_info)
        
        print(f"✅ Document uploaded successfully!")
        print(f"📊 Processed {len(chunks)} chunks from {len(text)} characters")
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")

def upload_youtube():
    """Upload and process YouTube video transcript"""
    url = input("🎥 Enter YouTube video URL: ").strip()
    
    try:
        # Get transcript
        text = get_youtube_transcript(url)
        
        if len(text.strip()) < 100:
            print("❌ Transcript too short or empty")
            return
        
        # Process text
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        
        # Save to database
        source_info = f"YouTube: {url}"
        save_to_database(chunks, embeddings, source_info)
        
        print(f"✅ YouTube transcript uploaded successfully!")
        print(f"📊 Processed {len(chunks)} chunks from {len(text)} characters")
        
    except Exception as e:
        print(f"❌ YouTube upload failed: {str(e)}")

# ========================================
# PART 8: MAIN MENU
# ========================================
# This is the main program entry point

def main():
    """Main program menu"""
    print("=" * 50)
    print("🤖 PDF/YOUTUBE TRANSCRIPT CHATBOT")
    print("=" * 50)
    
    while True:
        print("\n📋 MENU:")
        print("1. Upload PDF/TXT document")
        print("2. Upload YouTube video transcript")
        print("3. Start chatting")
        print("4. Check current database")
        print("5. Exit")
        
        choice = input("\n🎯 Choose option (1-5): ").strip()
        
        if choice == '1':
            upload_document()
        elif choice == '2':
            upload_youtube()
        elif choice == '3':
            chatbot_loop()
        elif choice == '4':
            db = load_database()
            if db:
                print(f"📚 Current database: {db['source_info']}")
                print(f"📊 Contains {len(db['chunks'])} chunks")
                print(f"📅 Created: {db.get('created_at', 'Unknown')}")
            else:
                print("❌ No database found")
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please choose 1-5.")

# ========================================
# PROGRAM START
# ========================================
if __name__ == "__main__":
    main()