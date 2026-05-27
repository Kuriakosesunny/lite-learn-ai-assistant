from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import StreamingResponse, FileResponse 
from fastapi.middleware.cors import CORSMiddleware 
from contextlib import asynccontextmanager 
from concurrent.futures import ThreadPoolExecutor 
import uvicorn, sqlite3, os, shutil, ollama, re, json, time 
import fitz, pytesseract 
from pdf2image import convert_from_path 
from PIL import Image, ImageEnhance
from services.faiss_service import add_embeddings, search

# Path to Tesseract OCR executable (required for scanned PDF processing) 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

# Create required folders if they don't exist 
for d in ("uploads", "slides"): os.makedirs(d, exist_ok=True)

# Words to ignore when extracting keywords from a student's question 
STOP_WORDS = { 
'what','is','how','explain','give','the','and','with','in','of','for','are', 
'its','about','define','describe','list','discuss','write','note','short', 
'example','types','mention','name','enumerate','a','an','to','do','using', 
'use','used','uses','between','difference','compare','which','why','when', 
'where','brief','detail','detailed','important','role','need','advantage' 
}

# Pattern to block non-academic queries like news, sports, weather, celebrity questions 
NON_ACADEMIC = re.compile( 
r'\b(who is|what is the name of).*(president|prime minister|pm|cm|ceo|founder)\b' 
r'|\b(current|latest|recent|today|news|2024|2025|2026)\b' 
r'|\b(cricket|football|movie|actor|actress|celebrity|song|music)\b' 
r'|\b(weather|temperature|stock|price|rate)\b', re.IGNORECASE)

# LLM parameter presets — controls context window, output length, and creativity 
# CHAT: short Q&A  |  LONG: detailed answers  |  XL: large lists  |  GEN: slides/maps  |  CARDS: flashcards/quiz 
LLM_CHAT  = {"num_ctx":3072,"num_predict":1536,"temperature":0.6,"top_p":0.9,"repeat_penalty":1.1} 
LLM_LONG  = {"num_ctx":4096,"num_predict":2048,"temperature":0.6,"top_p":0.9,"repeat_penalty":1.1} 
LLM_XL    = {"num_ctx":6144,"num_predict":2048,"temperature":0.6,"top_p":0.9,"repeat_penalty":1.1} 
LLM_GEN   = {"num_ctx":4096,"num_predict":2048,"temperature":0.7} 
LLM_CARDS = {"num_ctx":4096,"num_predict":1024,"temperature":0.7} 

# Opens a connection to the local SQLite database file 
def get_db(): 
    c = sqlite3.connect("litelearn.db", check_same_thread=False) 
    c.row_factory = sqlite3.Row 
    return c 

# Creates the 3 database tables on startup if they don't exist yet 
@asynccontextmanager 
async def lifespan(app): 
    db = get_db() 
    db.execute("CREATE TABLE IF NOT EXISTS textbooks (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT)") 
    db.execute("""
    CREATE TABLE IF NOT EXISTS rag_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        textbook_id INTEGER,
        chunk_text TEXT
    )
    """)
    db.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, textbook_id INTEGER, question TEXT, ai_response TEXT)") 
    db.commit(); yield 

app = FastAPI(lifespan=lifespan) 
# Allow the frontend (running on a different port) to make API requests 
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], 
allow_headers=["*"]) 
 
# Processes a single page image for OCR — enhances quality before running Tesseract 
def ocr_page(args): 
    num, img, total = args 
    try: 
        t0 = time.time() 
        w, h = img.size 
        # Upscale small images so Tesseract can read text more accurately 
        scale = 2.0 if w < 1000 else 1.5 if w < 1800 else 1.0 
        if scale > 1.0: img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS) 
        # Convert to grayscale, sharpen edges, boost contrast for better OCR accuracy 
        img = ImageEnhance.Contrast(ImageEnhance.Sharpness(img.convert('L')).enhance(2.0)).enhance(1.5) 
        # Run Tesseract: psm 6 = single block of text, oem 3 = best available engine 
        text = pytesseract.image_to_string(img, config='--psm 6 --oem 3') 
        # Remove non-ASCII characters and clean up extra whitespace 
        text = re.sub(r'[^\x00-\x7F]+',' ', text) 
        text = re.sub(r'\n{3,}','\n\n', re.sub(r'[ \t]{3,}',' ', text)) 
        text = re.sub(r'^\s*\d+\s*$','', text, flags=re.MULTILINE) 
        print(f"     Page {num+1}/{total} — {len(text)} chars ({time.time()-t0:.1f}s)") 
        return num, text 
    except Exception as e: 
        print(f"    Page {num+1}/{total}: {e}"); return num, "" 
 
# RAG scoring — scores each text chunk against the student's query 
# Higher score = more relevant chunk. Uses phrase, bigram, and keyword matching 
def score_chunks(chunks, words, bigrams, phrase, phrase_short=""): 
    scored = []
    for i, chunk in enumerate(chunks): 
        c = chunk.lower(); s = 0 
        # Full phrase match scores highest; bonus if found near the top of the chunk (likely a heading) 
        if phrase and phrase in c: 
            s += 80 + (20 if phrase in c[:400] else 0) 
        elif phrase_short and phrase_short in c: 
            s += 70 + (20 if phrase_short in c[:400] else 0) 
        # Bigram (2-word pair) matches boost score; heading matches get extra bonus 
        for bg in bigrams: 
            s += (15 if bg in c else 0) + (8 if bg in c[:400] else 0) 
        # Individual keyword matches — capped at 5 occurrences per word to avoid spam 
        s += sum(min(c.count(w), 5) * 2 for w in words) 
        scored.append((s, i)) 
    return sorted(scored, reverse=True) 
 
def is_reference_chunk(chunk: str) -> bool:
    """
    Determines whether the given chunk contains reference-style text (e.g., citations, bibliographies).
    Chunks with more than 3 citation-like markers are considered reference chunks.
    """
    # Define common citation signals
    ref_signals = len(re.findall(
        r'\b(pp?\.|vol\.|wiley|mcgraw|springer|proceedings|journal|isbn|doi|ed\.|eds\.|ibid|et al|[12]\d{3}\))',
        chunk, re.IGNORECASE))
    
    # Return True if more than 3 citation-like markers are found (indicating it's a reference chunk)
    return ref_signals >= 3

# Adjusting the context builder for dynamic chunking based on context length
def get_context(chunks, scored, n, force_top=False, available_context_len=0):
    if not scored or scored[0][0] < 2:
        return ""
    
    top = sorted(i for _, i in scored[:n])
    if available_context_len < 500:  # Small context, pull fewer chunks
        top = sorted(i for s, i in scored[:3])  # only top 3
    elif available_context_len < 1000:
        top = sorted(i for s, i in scored[:6])  # pull more if enough context

    extra = set()
    for _, bi in scored[:3]:
        for offset in range(-2, 5):
            ni = bi + offset
            if 0 <= ni < len(chunks) and ni not in top:
                if not is_reference_chunk(chunks[ni]):
                    extra.add(ni)

    selected = [i for i in sorted(set(top) | extra)
                if not is_reference_chunk(chunks[i])]
    return "\n\n---\n\n".join(chunks[i] for i in selected)
 
# Detects what kind of question the student is asking 
# This determines how many chunks to retrieve and what instructions to give the LLM 
def detect_qtype(q): 
    q = q.lower() 
    if any(w in q for w in ['compare','difference','vs','versus','contrast']): return 'comparison' 
    if any(w in q for w in ['define','definition','what is','what are']): return 'definition' 
    if re.search(r'\d+\s*marks?', q): return 'marks' 
    if any(w in q for w in ['explain','describe','discuss','elaborate']): return 'explain' 
    if any(w in q for w in ['types','list','enumerate','mention','name','steps','stages', 
                             'phases','components','features','advantages','disadvantages', 
                             'applications','characteristics','methods','techniques']): return 'list' 
    # Short bare-topic questions like "Stages of KDD" default to list type 
    if len(q.split()) <= 5: return 'list' 
    return 'general' 
 
# Update num_chunks to dynamically calculate based on context
def num_chunks(qtype, marks=0, available_context_len=0):
    if qtype in ('list', 'explain'):
        if available_context_len < 500:  # Less context, generate fewer points
            return 3
        elif available_context_len < 1000:  # Medium context, generate more
            return 6
        else:  # More context, generate a larger response
            return 8
    if qtype == 'marks': 
        return 3 if marks < 5 else 5
    return 4

# Adjust instruction dynamically
def instruction(qtype, marks, available_context_len):
    if available_context_len < 500:
        return (f"Answer briefly using the exact words and structure from the notes. "
                f"Cover a few points based on the notes. Stick to what is available in the context.")
    
    if available_context_len < 1000:
        return (f"Answer with sufficient details, covering as many points as possible from the notes. "
                f"Use the exact headings and terms as in the notes.")
    
    return (f"Answer using the exact terms from the notes. "
            f"Cover all aspects of the question using the available content. "
            f"Do not skip any point and do not add extra commentary.")
 
# Calls Mistral via Ollama and returns the full response as a string (non-streaming) 
def llm(messages, opts): 
    try: 
        return ollama.chat(model='mistral', messages=messages, options=opts, 
keep_alive=300)['message']['content'].strip() 
    except Exception as e: 
        print(f'LLM error: {e}') 
        return 'Sorry, the AI took too long to respond. Please try a shorter or simpler question.' 
 
# Calls Mistral via Ollama and yields tokens one by one (streaming mode for chat) 
def llm_stream(messages, opts): 
    try: 
        for chunk in ollama.chat(model='mistral', messages=messages, options=opts, stream=True, 
keep_alive=300): 
            yield chunk['message']['content'] 
    except Exception as e: 
        print(f'LLM stream error: {e}') 
        yield 'Sorry, the AI encountered an error. Please try again.' 
 
# Fetches the most recent chat Q&A for a textbook — used by flashcard, quiz, slides, mindmap 
def last_chat(db, tid): 
    return db.execute( 
        "SELECT question, ai_response FROM chat_history WHERE textbook_id=? ORDER BY rowid DESC LIMIT 1", 
        (tid,)).fetchone() 
 
# ── PDF UPLOAD 
@app.post("/api/pdf/upload") 
async def upload(file: UploadFile = File(...)): 
    # Save the uploaded file to the uploads folder 
    path = f"uploads/{file.filename}" 
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f) 
 
    # Step 1: Try direct text extraction using PyPDF2 (works for digital PDFs) 
    import fitz, pytesseract # Changed PyPDF2 to fitz (PyMuPDF)

    # Step 1: Try direct text extraction using PyMuPDF (much better for textbooks)
    text = "" 
    pages = 1 # Set a default value first
    try: 
        with fitz.open(path) as doc:
            pages = doc.page_count # Get the count while the document is OPEN
            for page in doc: 
                t = page.get_text() 
                if t: text += t + "\n" 
    except Exception as e: 
        print(f"PyMuPDF error: {e}") 

    chars = len(text.strip()) 
    # The 'pages' variable now safely holds the number even after doc closes
 
    # Step 2: If extracted text is too short, it's likely a scanned PDF — use OCR instead 
    if chars < 500 or chars // max(1, pages) < 100: 
        t_ocr = time.time() 
        print("     Rendering PDF to images...") 
        try: 
            imgs = convert_from_path(path, dpi=150, use_pdftocairo=True) 
        except: 
            imgs = convert_from_path(path, dpi=150) 
        total = len(imgs) 
        print(f"            {total} pages rendered in {time.time()-t_ocr:.1f}s — starting OCR...") 
        results = [""] * total 
        # Run OCR on all pages in parallel using 4 threads for speed 
        with ThreadPoolExecutor(max_workers=4) as ex: 
            for n, t in ex.map(ocr_page, [(n, img, total) for n, img in enumerate(imgs)]): 
                results[n] = t 
        text = "\n".join(results) 
        print(f"   OCR done: {len(text)} chars in {time.time()-t_ocr:.1f}s") 
 
    if not text.strip(): return {"error": "Could not extract text. Try a cleaner scan."} 
 
    # Step 3: Clean the text — remove non-ASCII, excess spaces and blank lines 
    text = re.sub(r'[^\x00-\x7F]+',' ', re.sub(r'[ \t]{3,}',' ', re.sub(r'\n{4,}','\n\n', text))).strip() 

    print("PDF upload received")
 
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # ... inside upload endpoint ...
    # Step 4: Use semantic chunking to avoid cutting sentences/words in half 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text) 

    print("Chunks:", len(chunks))
 
    # Step 5: Save textbook record and all chunks into SQLite 
    db = get_db() 
    db.execute("INSERT INTO textbooks (filename) VALUES (?)", (file.filename,)) 
    tid = db.execute("SELECT last_insert_rowid()").fetchone()[0] 
    chunk_ids = []

    for c in chunks:
        cur = db.execute(
            "INSERT INTO rag_chunks (textbook_id, chunk_text) VALUES (?,?)",
            (tid, c)
        )
        chunk_ids.append(cur.lastrowid)

    db.commit()
    # build FAISS embeddings
    add_embeddings(chunks, chunk_ids)
    print("Embeddings added to FAISS")
    print(f"   Uploaded: '{file.filename}' | ID={tid} | Chunks={len(chunks)}") 
    return {"id": tid} 
 
# Returns the list of all uploaded textbooks 
@app.get("/api/textbooks") 
async def get_books(): 
    return [dict(r) for r in get_db().execute("SELECT * FROM textbooks").fetchall()] 
 
# ── CHAT 
@app.post("/api/chat") 
async def chat(data: dict): 
    tid, q = data["textbook_id"], data["question"].strip() 
 
    # Block non-academic questions (news, sports, weather, etc.) 
    if NON_ACADEMIC.search(q): 
        return StreamingResponse(iter(["I answer only from your uploaded notes.\nTry: Explain supervised learning"]), media_type="text/plain") 
 
    db = get_db() 
    qtype = detect_qtype(q) 
    marks = int(m.group(1)) if (m := re.search(r'(\d+)\s*marks?', q.lower())) else 0 
    item_count = 1  # Default; updated below after counting items in context 
 
    # Extract meaningful keywords and bigrams from the question for chunk scoring 
    q_clean = re.sub(r'[^\w\s]','', q.lower()) 
    words   = [w for w in q_clean.split() if len(w) > 2 and w not in STOP_WORDS] 
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)] 
 
    # Build the phrase used for full-phrase matching (strip question-type words) 
    PHRASE_STOP = {'explain','describe','discuss','elaborate','define','list','what','how', 
                   'give','write','mention','enumerate','name','steps','example','illustrate'} 
    phrase       = ' '.join(w for w in q_clean.split() if w not in PHRASE_STOP and len(w) > 1) 
    phrase_short = ' '.join(words) 
 
    # Load last 6 chat messages for context (used in example follow-up detection) 
    history = list(reversed(db.execute( 
        "SELECT question, ai_response FROM chat_history WHERE textbook_id=? ORDER BY id DESC LIMIT 6", 
        (tid,)).fetchall())) 
 
    # Check if student is asking for examples of the previous answer 
    is_example = bool(re.search(r'\b(examples?|illustrate|instances?)\b', q, re.IGNORECASE)) and history 
    if is_example: 
        # Use the previous Q&A as context instead of searching chunks 
        last_q, last_a = history[-1][0], history[-1][1] 
        ctx    = f"PREVIOUS QUESTION: {last_q}\n\nPREVIOUS ANSWER:\n{last_a}" 
        qtype  = "example" 
        source = "examples_followup" 
        print(f"[SOURCE: examples_followup] prev_q={last_q!r}") 
    else: 
        # Load all chunks for this textbook and score them against the query 
        ids = search(q, k=30)
        print("FAISS IDs:", ids)

        if not ids:
            return StreamingResponse(iter(["No relevant content found."]), media_type="text/plain")

        placeholders = ",".join(["?"] * len(ids))

        rows = db.execute(
            f"SELECT chunk_text FROM rag_chunks WHERE id IN ({placeholders}) AND textbook_id=?",
            (*ids, tid)
        ).fetchall()
        print("Rows found:", len(rows))

        chunks = [r[0] for r in rows]
        if not chunks: 
            return StreamingResponse(iter(["No content found. Please re-upload your PDF."]), media_type="text/plain") 
        scored = score_chunks(chunks, words, bigrams, phrase, phrase_short) 
        force  = qtype in ('list','explain') 
        ctx    = get_context(chunks, scored, num_chunks(qtype, marks), force_top=force) 
        source = "notes" 
        print(f"[SOURCE: notes] type={qtype} marks={marks} phrase={phrase!r} ctx_chars={len(ctx)}") 
 
    # Build the prompt based on source type 
    if not ctx: 
        # No relevant chunks found — fall back to general knowledge 
        prefix = "Not found in notes — using general knowledge:\n\n" 
        prompt = f"Answer clearly in bullet points: {q}" 
    elif source == "examples_followup": 
        prefix = "" 
        prompt = (f"You are a study assistant.\n\n{ctx}\n\n" 
                  "Give real-world examples for EACH item in the previous answer.\n" 
                  "Format: **Name**\nExample: [text]\nCover ALL items. Do not skip.") 
    else: 
        # Main RAG prompt — grounded strictly in the retrieved notes 
        topic = re.sub(r'\d+\s*marks?','', re.sub(r'\b(explain|describe|discuss)\b','', q, flags=re.IGNORECASE)).strip() 
        prefix = "" 
        # Removed the regex item counting to prevent forced hallucinations
        prompt = (f"You are a study assistant. Answer ONLY from the NOTES provided.\n\n" 
                  f"NOTES:\n{ctx}\nEND NOTES\n\n" 
                  f"QUESTION: {q}\n\n" 
                  f"RULES:\n" 
                  f"1. Use ONLY the NOTES. Zero outside knowledge. Do NOT use anything from your training.\n" 
                  f"2. Extract and explain all relevant points, steps, or features found in the NOTES regarding [{topic}]. Do not leave out any critical steps mentioned in the notes.\n"
                  f"3. SKIP anything in the notes that is NOT about [{topic}].\n" 
                  f"4. Use the EXACT stage/point names from the NOTES as headings. Explain each one simply in your own words — easy to understand, but do NOT rename, merge, or skip any of them.\n" 
                  f"5. {instruction(qtype, marks, len(ctx))}\n"
                  f"6. Do NOT add examples unless the notes contain them for this topic.\n" 
                  f"7. Write the complete answer. Do NOT stop early. Do NOT skip any stage or point.") 
 
    # Choose LLM preset based on question complexity 
    t0 = time.time() 
    lm = (LLM_XL   if qtype in ('list','explain') and item_count >= 6 else 
          LLM_LONG if qtype in ('list','explain') or marks >= 10 else LLM_CHAT) 
 
    # Stream the response token by token — saves to DB after stream completes 
    def generate(): 
        buf = [prefix] if prefix else [] 
        if prefix: yield prefix 
        for token in llm_stream([{'role':'user','content':prompt}], lm): 
            buf.append(token) 
            yield token 
        answer = ''.join(buf) 
        print(f"        Chat: {time.time()-t0:.1f}s | {len(answer.split())} words") 
        db.execute("INSERT INTO chat_history (textbook_id,question,ai_response) VALUES (?,?,?)", (tid, q, answer)) 
        db.commit() 
 
    return StreamingResponse(generate(), media_type="text/plain") 
 
# Returns full chat history for a textbook (used to restore chat on page reload) 
@app.get("/api/chat-history") 
async def get_chat_history(textbook_id: int): 
    rows = get_db().execute( 
        "SELECT question, ai_response FROM chat_history WHERE textbook_id=? ORDER BY id", 
        (textbook_id,)).fetchall() 
    return {"history": [{"q": r[0], "a": r[1]} for r in rows]} 
 
# Deletes all chat messages for a textbook 
@app.post("/api/clear-chat") 
async def clear_chat(data: dict): 
    db = get_db() 
    db.execute("DELETE FROM chat_history WHERE textbook_id=?", (data["textbook_id"],)) 
    db.commit() 
    return {"status": "cleared"} 



# ── Flashcards ───────────────────────────────────────────────────────────────────────
from json_repair import repair_json
@app.post("/api/generate-cards")
async def gen_cards(data: dict):
    tid, count = data["textbook_id"], int(data.get("count", 5))
    row = last_chat(get_db(), tid)
    if not row: return {"error":"Ask a question in Chat first!"}
    prompt = (
    f"Topic: {row[0]}\n"
    f"Content:\n{row[1]}\n\n"
    f"Create EXACTLY {count} flashcards.\n"
    f"Each flashcard must contain ONE concept.\n"
    f"The JSON array MUST contain exactly {count} objects.\n\n"
    'Return ONLY a JSON array like this:\n'
    '[{"t":"Term","f":"Definition"}]'
)
    t0  = time.time()
    raw = llm([{'role':'user','content':prompt}], LLM_CARDS)\
        .replace("```json","").replace("```","").strip()

    print("\n===== RAW FLASHCARD RESPONSE =====")
    print(raw)
    print("==================================\n")
    print(f"\u23f1\ufe0f  Flashcards: {time.time()-t0:.1f}s")
    try:
        parsed = json.loads(repair_json(raw))

        # If AI returned {"data":[...]}
        if isinstance(parsed, dict) and "data" in parsed:
            cards = parsed["data"]

        # If AI returned [...]
        elif isinstance(parsed, list):
            cards = parsed

        else:
            raise ValueError("Unexpected JSON format")
        cards = cards[:count]  # enforce max count

    except Exception as e:
        print("FLASHCARD PARSE ERROR:", e)
        print("RAW RESPONSE:", raw)
        return {"error":"AI failed to format cards. Try again."}
    return {"cards": cards}

# ── Quiz ────────────────────────────────────────────────────────────────────────────────
@app.post("/api/generate-quiz")
async def gen_quiz(data: dict):
    tid, count, level = data["textbook_id"], int(data.get("count",3)), data.get("level","Medium")
    row = last_chat(get_db(), tid)
    if not row: return {"error":"Ask a question in Chat first!"}
    prompt = (
        f"Topic: {row[0]}\nContent:\n{row[1]}\n\n"
        f"Create EXACTLY {count} {level}-level MCQ questions from the content.\n"
        "Each question must have 4 options and 1 correct answer.\n\n"
        "IMPORTANT:\n"
        "- Options must be plain text.\n"
        "- DO NOT include A), B), C), D) in the options.\n\n"
        'Return ONLY valid JSON in this format:\n'
        '[{"q":"Question","o":["Option 1","Option 2","Option 3","Option 4"],"c":"Correct Option Text"}]'
    )
    t0  = time.time()
    raw = llm([{'role':'user','content':prompt}], LLM_GEN).replace("```json","").replace("```","").strip()
    print(f"\u23f1\ufe0f  Quiz: {time.time()-t0:.1f}s")
    try:
        quiz = json.loads(re.search(r'\[.*\]', raw, re.DOTALL).group())
        
        for q in quiz:
            q["o"] = [re.sub(r'^[A-D][\.\)]\s*', '', opt).strip() for opt in q["o"]]
    except: return {"error":"AI failed to format quiz. Try again."}
    return {"data": quiz}


# ── Slides ────────────────────────────────────────────────────────────────────
@app.post("/api/generate-slides")
async def gen_slides(data: dict):
    from pptx import Presentation
    from pptx.util import Pt, Inches
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.oxml.ns import qn
    from lxml import etree

    tid, topic, name = data["textbook_id"], data.get("topic","Overview"), data.get("student_name","Student")
    row = last_chat(get_db(), tid)
    if not row: return {"error": "Ask a question in Chat first!"}

    prompt = (f"Topic: {topic}\nContent:\n{row[1]}\n\n"
              "Create 6 educational slides, 5-6 bullet points each, full sentences.\n"
              "Do NOT put 'Slide 1', 'Slide 2' etc in titles. Complete all 6 slides.\n"
              'Return ONLY JSON: [{"title":"Topic Name","content":["Sentence 1.","Sentence 2."]}]')
    t0  = time.time()
    raw = llm([{'role':'user','content':prompt}], LLM_GEN).replace("```json","").replace("```","").strip()
    try: slides_data = json.loads(re.search(r'\[.*\]', raw, re.DOTALL).group())
    except: return {"error": "AI failed to format slides. Try again."}
    print(f"  ⏱️  Slides: {time.time()-t0:.1f}s")

    RED = RGBColor(0xCC,0x00,0x01); WHT = RGBColor(0xFF,0xFF,0xFF)
    BLK = RGBColor(0x22,0x22,0x22); GRY = RGBColor(0x55,0x55,0x55)
    TNR = "Times New Roman"

    def bg(sl, c):
        f = sl.background.fill; f.solid(); f.fore_color.rgb = c

    def box(sl, l, t, w, h, c):
        s = sl.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
        s.fill.solid(); s.fill.fore_color.rgb = c; s.line.fill.background()

    def txt(sl, s, l, t, w, h, sz=28, bold=False, col=BLK, align=PP_ALIGN.LEFT, italic=False):
        tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = align; r = p.add_run(); r.text = s
        r.font.size = Pt(sz); r.font.bold = bold; r.font.color.rgb = col
        r.font.italic = italic; r.font.name = TNR

    prs = Presentation(); prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
    SW, SH = 13.33, 7.5

    # Title slide
    ts = prs.slides.add_slide(prs.slide_layouts[6]); bg(ts, WHT)
    box(ts, 0, 0, SW, 0.18, RED)
    box(ts, 0, SH-0.18, SW, 0.18, RED)
    txt(ts, topic.upper(), 0.6, 1.8, SW-1.2, 2.0, sz=40, bold=True, col=RED, align=PP_ALIGN.CENTER)
    box(ts, 2.5, 3.95, SW-5.0, 0.04, RED)
    txt(ts, f"Prepared by:  {name}", 0.6, 4.1, SW-1.2, 0.6, sz=28, col=GRY, italic=True, align=PP_ALIGN.CENTER)
    txt(ts, "LiteLearn  •  AI Study Assistant", 0.6, 4.7, SW-1.2, 0.5, sz=22, col=RGBColor(0xAA,0xAA,0xAA), align=PP_ALIGN.CENTER)

    # Content slides
    for i, s in enumerate(slides_data[:6]):
        sl = prs.slides.add_slide(prs.slide_layouts[6]); bg(sl, WHT)
        box(sl, 0, 0, SW, 0.18, RED)
        box(sl, 0, SH-0.18, SW, 0.18, RED)
        pn = sl.shapes.add_textbox(Inches(SW-0.8), Inches(SH-0.17), Inches(0.65), Inches(0.15))
        pn.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
        r = pn.text_frame.paragraphs[0].add_run()
        r.text = str(i+2); r.font.size = Pt(11); r.font.color.rgb = WHT; r.font.bold = True; r.font.name = TNR
        title = re.sub(r'^slide\s*\d+\s*[:\-–]?\s*','', s.get('title',''), flags=re.IGNORECASE).strip()
        ht = sl.shapes.add_textbox(Inches(0.45), Inches(0.28), Inches(SW-0.9), Inches(0.85))
        ht.text_frame.word_wrap = True
        hp = ht.text_frame.paragraphs[0]; hp.alignment = PP_ALIGN.LEFT
        hr = hp.add_run(); hr.text = title; hr.font.size = Pt(40); hr.font.bold = True
        hr.font.color.rgb = BLK; hr.font.name = TNR
        box(sl, 0.45, 1.18, SW-0.9, 0.04, RED)
        points = s.get('content',[]); n = len(points)
        ct = sl.shapes.add_textbox(Inches(0.45), Inches(1.28), Inches(SW-0.9), Inches(5.9))
        tf = ct.text_frame; tf.word_wrap = True
        font_sz = 28 if n <= 4 else (24 if n <= 6 else 20)
        total_pt = 5.9 * 72
        per_item = min(total_pt / max(n, 1), 110)
        space_before_pt = max(4, per_item - font_sz * 1.4 - 4)
        for j, pt in enumerate(points):
            p = tf.add_paragraph() if j > 0 else tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            pPr = p._p.get_or_add_pPr()
            lnSpc = etree.SubElement(pPr, qn('a:lnSpc'))
            spcPct = etree.SubElement(lnSpc, qn('a:spcPct'))
            spcPct.set('val','100000')
            if j > 0:
                spaBef = etree.SubElement(pPr, qn('a:spcBef'))
                spcPts = etree.SubElement(spaBef, qn('a:spcPts'))
                spcPts.set('val', str(int(space_before_pt * 100)))
            d = p.add_run(); d.text = "•  "; d.font.size = Pt(font_sz); d.font.color.rgb = RED; d.font.name = TNR
            r = p.add_run(); r.text = str(pt); r.font.size = Pt(font_sz); r.font.color.rgb = BLK; r.font.name = TNR

    # Thank You slide
    ty = prs.slides.add_slide(prs.slide_layouts[6]); bg(ty, WHT)
    box(ty, 0, 0, SW, 0.18, RED)
    box(ty, 0, SH-0.18, SW, 0.18, RED)
    txt(ty, "Thank You!", 0.6, 2.2, SW-1.2, 1.5, sz=54, bold=True, col=RED, align=PP_ALIGN.CENTER)
    box(ty, 2.5, 3.85, SW-5.0, 0.04, RED)
    txt(ty, f"Presented by:  {name}", 0.6, 4.0, SW-1.2, 0.6, sz=28, col=GRY, italic=True, align=PP_ALIGN.CENTER)
    txt(ty, "Keep Learning  |  LiteLearn", 0.6, 4.6, SW-1.2, 0.5, sz=22, col=RGBColor(0xAA,0xAA,0xAA), align=PP_ALIGN.CENTER)

    os.makedirs("slides", exist_ok=True)
    out = f"slides/{topic.replace(' ','_')}.pptx"; prs.save(out)
    return FileResponse(out, filename=f"{topic}.pptx")

@app.post("/api/generate-mindmap")
async def gen_mindmap(data: dict):
    tid = data["textbook_id"]
    row = last_chat(get_db(), tid)
    if not row:
        return {"error": "Ask a question in Chat first!"}
 
    raw_topic = row[0].strip()
    topic = re.sub(
        r'^(explain|describe|discuss|define|list|what is|what are|write about|elaborate on)\s+',
        '', raw_topic, flags=re.IGNORECASE
    ).strip()[:80]
    safe  = re.sub(r'\b(and|its|the|of|in|for|with|a|an)\b', '', topic, flags=re.IGNORECASE)
    safe  = re.sub(r'\s{2,}', ' ', safe).strip().replace('"', "'")
    context = row[1][:3000]
 
    prompt = (
        f"Topic: {topic}\nFull Answer:\n{context}\n\n"
        "Create a Mermaid.js mind map covering ALL main points in the answer.\n"
        "STRICT RULES:\n"
        f'1. FIRST line after graph LR MUST be: ROOT["{safe}"]\n'
        "2. Each main section = one branch from ROOT\n"
        "3. Sub-details connect from their parent branch\n"
        "4. Use IDs: ROOT, A, B, C, D, E, F, G, H\n"
        "5. Labels: double-quoted, max 4 words, no punctuation\n"
        "6. Use --> only. Wrap in ```mermaid\n\n"
        "Example:\n```mermaid\ngraph LR\n"
        f'ROOT["{safe}"] --> A["Finance"]\n'
        'ROOT --> B["HR Management"]\n'
        'A --> C["Forecasting"]\n```'
    )
    t0   = time.time()
    resp = llm([{'role': 'user', 'content': prompt}], LLM_CARDS)
    print(f"  ⏱️  Mindmap: {time.time()-t0:.1f}s")
 
    m    = re.search(r'```mermaid\s*(.*?)\s*```', resp, re.DOTALL | re.IGNORECASE)
    code = m.group(1).strip() if m else resp.strip()
    code = code.replace(" -- ", " --> ")
    # Strips both single and double quotes from the label before applying the required double quotes
    code = re.sub(
    r'(\w+)\[([^\]]+)\]', 
    lambda x: f'{x.group(1)}["{x.group(2).strip().strip("\"\'")}"]', 
    code
)
    if not re.match(r'^graph\s', code, re.IGNORECASE):
        code = "graph LR\n" + code
 
    lines  = code.split('\n')
    header = [l for l in lines if re.match(r'^graph\s', l.strip(), re.IGNORECASE)]
    edges  = [l for l in lines if '-->' in l][:12]
    clean_edges = []

    for l in edges:
        parts = l.split('-->')
        if len(parts) != 2:
            continue

        src = re.match(r'\s*(\w+)', parts[0])
        tgt = re.match(r'\s*(\w+)', parts[1].strip())

        if not src or not tgt:
            continue

        # remove self loops like ROOT-->ROOT or A-->A
        if src.group(1) == tgt.group(1):
            continue

        # remove anything pointing TO root
        if tgt.group(1) == "ROOT":
            continue

        clean_edges.append(l)

    edges = clean_edges
    edges = [l for l in edges if not re.search(r'\bROOT\b\s*-->\s*\bROOT\b', l)]
 
    edges = [re.sub(r'\bROOT\["?ROOT"?\]',          f'ROOT["{safe}"]', l) for l in edges]
    edges = [re.sub(r'\bROOT\["?[A-Z][A-Z\s]+"?\]', f'ROOT["{safe}"]', l) for l in edges]
 
    def should_remove(line):
        parts = line.split('-->')
        if len(parts) != 2: return False
        src = re.match(r'\s*(\w+)', parts[0])
        tgt = re.match(r'\s*(\w+)', parts[1].strip())
        if not src or not tgt: return False
        return src.group(1) == tgt.group(1) or tgt.group(1) == 'ROOT'
 
    edges = [l for l in edges if not should_remove(l)]
 
    if not any(re.match(r'\s*ROOT', l) for l in edges) and edges:
        first = re.match(r'\s*(\w+)', edges[0])
        if first:
            edges.insert(0, f'ROOT["{safe}"] --> {first.group(1)}')
 
    diagram_lines = header + [f'ROOT["{safe}"]'] + edges
    return {"data": '\n'.join(diagram_lines)}



@app.get("/api/debug-chunks")
async def debug_chunks(textbook_id: int, keyword: str = ""):
    chunks = [r[0] for r in get_db().execute("SELECT chunk_text FROM rag_chunks WHERE textbook_id=?",(textbook_id,))]
    if keyword:
        hits = [(i,c[:300]) for i,c in enumerate(chunks) if keyword.lower() in c.lower()]
        return {"total":len(chunks),"matches":len(hits),"results":hits}
    return {"total":len(chunks),"preview":[c[:200] for c in chunks[:5]]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)