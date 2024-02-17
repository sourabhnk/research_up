import io
from transformers import BartTokenizer, BartForConditionalGeneration
import streamlit as st
import PyPDF2
import re


def get_pdf_reader(uploaded_file):
    reader = None
    if uploaded_file is not None:
        file_bytes = io.BytesIO(uploaded_file.read())
        reader = PyPDF2.PdfReader(file_bytes)
    return reader


def extract_text_from_page(page):
    text = page.extract_text()
    return text if text else ""


def find_sections_with_heuristics(pdf_reader):
    introduction = []
    methodology = []
    conclusion = []

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = extract_text_from_page(page).lower()

        # Define some heuristics for finding sections
        if re.search(r'\b(introduction|abstract)\b', text):
            introduction.append(page_num)
        elif re.search(r'\b(methods|methodology)\b', text):
            methodology.append(page_num)
        elif re.search(r'\b(conclusion|conclusions|summary|discussion)\b', text):
            conclusion.append(page_num)

    # Return the first match of each section, assuming that once it starts, it continues until the next one begins.
    sections = {
        "Introduction": introduction[0] if introduction else None,
        "Methodology": methodology[0] if methodology else None,
        "Conclusion": conclusion[0] if conclusion else None
    }

    return sections


def generate_summary(context):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer([context], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = summarization_model.generate(inputs['input_ids'], num_beams=4, length_penalty=2.0, max_length=400, min_length=60, no_repeat_ngram_size=3)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def extract_and_summarize_section(pdf_reader, start_page, end_page=None):
    if start_page is None:
        return None

    end_page = end_page or len(pdf_reader.pages)
    text = ""

    for page_num in range(start_page, end_page):
        page = pdf_reader.pages[page_num]
        text += extract_text_from_page(page)

    if text.strip():
        return generate_summary(text)

    return None


def main():
    st.title("Research Paper Interpreter Bot")

    uploaded_file = st.file_uploader("Upload a research paper, max 20MB", type=["pdf"], accept_multiple_files=False)

    if uploaded_file is not None:
        pdf_reader = get_pdf_reader(uploaded_file)
        sections = find_sections_with_heuristics(pdf_reader)

        if st.button("Summarize Sections"):
            with st.spinner("Generating summaries...this may take a while"):

                intro_summary = extract_and_summarize_section(
                    pdf_reader,
                    sections["Introduction"],
                    sections.get("Methodology")
                )

                method_summary = extract_and_summarize_section(
                    pdf_reader,
                    sections["Methodology"],
                    sections.get("Conclusion")
                )

                conclusion_summary = extract_and_summarize_section(
                    pdf_reader,
                    sections["Conclusion"]
                )

                if intro_summary:
                    st.header("Introduction Summary:")
                    st.write(intro_summary)

                if method_summary:
                    st.header("Methodology Summary:")
                    st.write(method_summary)

                if conclusion_summary:
                    st.header("Conclusion Summary:")
                    st.write(conclusion_summary)


if __name__ == "__main__":
    main()
