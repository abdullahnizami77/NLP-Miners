import streamlit as st
from pdfminer.high_level import extract_text
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text
from haystack import Document
from haystack.nodes import PreProcessor
from sentence_transformers import SentenceTransformer
import faiss
from gradio_client import Client

client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")

# Streamlit App


# PDF Processing and Indexing Functions
def process_pdf_and_create_index(index_path):
    # Extract text from PDF
   # pdf_text = extract_text(pdf_file_path)
    pdf_text=" "
    btl_text = ""
    with open("./raw data.txt", 'r', encoding = 'utf-8') as f:
      btl_text = f.read()
    pdf_text+=btl_text

    # Initialize a DocumentStore
    document_store = InMemoryDocumentStore()

    # Clean the text
    cleaned_text = clean_wiki_text(pdf_text)

    # Create Haystack Document snippets
    preprocessor = PreProcessor(split_length=300, split_overlap=1,
                                split_respect_sentence_boundary=True,
                                clean_empty_lines=True, clean_whitespace=True,
                                clean_header_footer=True)
    snippets = preprocessor.process(Document(content=cleaned_text))

    # Extract snippet content
    snippet_text = [snippet.content for snippet in snippets]

    # Generate embeddings of snippets
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(snippet_text)

    # Create FAISS vector index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return snippet_text

def process_and_query(index_path, snippet_text, query_text, k=5):
    index = faiss.read_index(index_path)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_embedding = model.encode([query_text])

    #query_embedding = model.encode([query_text])

    D, I = index.search(query_embedding, k)

    nearest_neighbors_data = []
    for idx in I[0]:
        nearest_neighbors_data.append(snippet_text[idx])

    return nearest_neighbors_data

def get_resp(prompt):
  result = client.predict(
				prompt,	# str in 'parameter_28' Textbox component
				prompt,	0,0,0,1,	api_name="/chat_1"
  )
  output_text= result

    return output_text
def main():
    st.markdown(
        """
        <style>
        .stApp {
         background: radial-gradient(ellipse at center, rgba(255, 255, 255, 0.95) 10%, rgba(0, 0, 0, 0.6) 100%);;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <style>
        .custom-content {
        border: 2px solid #333; /* Change to your preferred border color */
        padding: 10px;
        border-radius: 10px; /* Adjust border radius for rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='text-align:center'>CHATBOT FOR MINISTRY OF COAL</h1>", unsafe_allow_html=True)

    st.markdown('<div class="custom-content">', unsafe_allow_html=True)

    st.markdown("<p style='text-align:center'>Prototype developed by Team NLP MINERS for SIH 2023</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


    index_path ='index_file.index'
    snippet_text = process_pdf_and_create_index(index_path)
    k=5

    question = st.text_input("Enter your question")
    query_text = question
    nearest_neighbors = process_and_query(index_path, snippet_text, query_text, k)
    knowledge_context = '\n'.join(nearest_neighbors)
    question = "GIVEN QUESTION : \n" + question
    knowledge_context = " RELEVENT KNOWLEDGE CONTEXT: \n"+ knowledge_context
    context = """
    above is a question regarding coal laws and relevant knowledge snippets from acts, ammendment and laws realted to question ,
    please use context/snippet and provide simplifed answer to the user"""


    prompt= question+ knowledge_context + "\n"*3 + "\n"*3 + context
    mic_container = st.markdown("<div class='centered'>", unsafe_allow_html=True)
    mic_icon = "microphone-icon.png"  # Replace with the path to your microphone icon image
    st.image(mic_icon, use_column_width=False, width=30)
    st.markdown("Use voice as input to interact with our chatbot", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    if st.button("Answer"):
        response = get_resp(prompt)
        st.text("Answer:")
        st.write(response)
        #st.write(prompt)
            #st.write(knowledge_context)
            #st.write(query_text)

if __name__ == "__main__":
    main()
