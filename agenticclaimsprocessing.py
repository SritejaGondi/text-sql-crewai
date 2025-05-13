import os
import io
import re
import base64
import json
import tempfile
import streamlit as st
import boto3
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional, Type
from pydantic import BaseModel, validator
import pandas as pd
import concurrent.futures
import uuid
import subprocess
import logging

# LangChain imports for OCR processing and QA chain
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.chat_models import init_chat_model
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

# Import RecursiveCharacterTextSplitter from langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Together AI Python client for direct image processing
from together import Together

# Import CrewAI classes (adjust the import path as needed)
from crewai import LLM
from crewai import Agent, Task, Crew, Process

from supabase import create_client, Client

# --- AWS Credentials (configure appropriately) ---
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""

# --- Supabase Client Configuration ---
SUPABASE_URL = ""
SUPABASE_KEY = ""
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Set up the Streamlit page ---
st.set_page_config(page_title="Unified Document Processor", layout="wide")
st.title("Unified Document Processor with Concurrent Textract OCR & LLM Vision")

# --- Together AI LLM Configuration ---
st.header("LLM Configuration (Together AI)")
together_api_key = st.text_input("Enter your Together AI API Key", type="password")
together_model = st.text_input("Enter Together AI Model Name", value="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")

if st.button("Configure Model", key="configure_model"):
    if not together_api_key or not together_model:
        st.error("Please provide both API key and model name!")
    else:
        os.environ["TOGETHER_API_KEY"] = together_api_key
        crewllm = LLM(model=f"together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", temperature=0.7)
        lcllm = init_chat_model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier", streaming=True, temperature=0.7, model_provider="together", api_key=together_api_key)
        st.session_state["crewllm"] = crewllm
        st.session_state["lcllm"] = lcllm
        st.success(f"VLM Extraction configured model: {together_model}")
        st.success(f"CrewAI configured LLM: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
        st.success(f"OCR Langchain LLM configured: deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")

# --- Document Source Input ---
st.header("Document Source")
st.info("Either upload a document file OR specify an S3 bucket (with optional prefix) below. If a file is uploaded, it takes precedence.")
uploaded_file = st.file_uploader("Upload a document (PDF, PNG, JPG, JPEG)", type=["pdf", "png", "jpg", "jpeg"])
s3_bucket = st.text_input("Or enter S3 Bucket Name")
s3_prefix = st.text_input("Enter S3 Bucket Prefix (optional)")

# --- Query Input ---
st.header("Enter Your Query")
query = st.text_area("Enter your question or prompt", value="Extract all the information you can find in the claims. Process and review the claims.")

#############################
# Helper Function: List S3 Files
#############################
def list_s3_files(bucket, prefix=""):
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    files = []
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                    files.append(key)
    return files

#############################
# Helper Function: Get S3 File
#############################
def get_s3_file(bucket, key):
    s3_client = boto3.client("s3")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_data = response["Body"].read()
        return file_data, key
    except Exception as e:
        st.error(f"Error downloading {key} from S3: {e}")
        return None, key

#############################
# Helper Function: Get Local File Data
#############################
def get_file_data():
    file_data = None
    file_name = ""
    if uploaded_file:
        file_name = uploaded_file.name.lower()
        file_data = uploaded_file.getvalue()
    return file_data, file_name

#############################
# Recursive Text Splitter Function
#############################
def split_and_parse_raw_output(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        length_function=len,
        is_separator_regex=False,  # literal separators
    )
    docs = text_splitter.create_documents([raw_text])
    extracted = {}
    pattern = r'\n\t\*\s*\*\*\n\n\t'
    for doc in docs:
        for line in doc.page_content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                extracted[key] = value
    return extracted

#############################
# Field Parsing Function
#############################
def parse_claim_data(raw_text, method, file_name):
    split_data = split_and_parse_raw_output(raw_text)
    return {
        "file_name": file_name,
        "Method": method,
        "parsed_fields": split_data,
        "raw_output": raw_text
    }

#############################
# Custom Vision Tool
#############################
class ImagePromptSchema(BaseModel):
    image_path_url: str

    @validator("image_path_url")
    def validate_image_path_url(cls, v: str) -> str:
        if v.startswith("http") or v.startswith("data:image"):
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {v}")
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Unsupported image format. Supported formats: {valid_extensions}")
        return v

class VisionTool:
    name = "Vision Tool"
    description = "This tool uses Together AI's Vision API (Llama vision model) to describe the contents of an image."
    args_schema: Type[BaseModel] = ImagePromptSchema
    _client: Optional[Together] = None

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = Together(api_key=api_key)

    def run(self, image_path_url: str) -> str:
        ImagePromptSchema(image_path_url=image_path_url)
        if not (image_path_url.startswith("http") or image_path_url.startswith("data:image")):
            with open(image_path_url, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_data = f"data:image/jpeg;base64,{base64_image}"
        else:
            image_data = image_path_url
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ]}
            ]
        )
        return response.choices[0].message.content

#############################
# Pydantic Structured Output
#############################
class DocumentReviewOutput(BaseModel):
    file_name: str
    vendor: str
    service_type: str
    claim_amt: float
    claim_date: str
    claim_id: str
    claim_type: str
    decision: str
    explanation: str

#############################
# CrewAI Document Review Function
#############################
def review_document_with_crewai():
    review_agent = Agent(
        role="Health Insurance Document Reviewer for document {id}",
        goal="Review the following extracted data from the document {id}: {extracted_text}. Analyze key details including the type of document, content, summary and any other pertinent details.",
        backstory="You are an agent specialized in understanding data in the document {id} : {extracted_text}; found in health insurance claim documents. Your job is to categorize the claim type (e.g., medical, insurance, address change, etc..) and analyze important details to assist in making an informed decision.",
        memory=True,
        llm=st.session_state["crewllm"],
        #max_rpm=6,
        respect_context_window=True,
        #cache=True,
        verbose=True
    )
    claim_agent = Agent(
        role="Claim Decision Maker using business rules on the health insurance document {id}.",
        goal="Make a decision based on the provided data from document {id}: {extracted_text}; using business process rules. Output whether the claim is APPROVED, PARTIALLY APPROVED, INELIGIBLE, or PENDING for further human review. Provide a brief explanation on the output decision.",
        backstory="You are an expert in health insurance claims with deep understanding of business rules. You use the business rules as reference to review the document {id}'s extracted data: {extracted_text}; and determine whether the claim should be APPROVED, PARTIALLY APPROVED, INELIGIBLE, or PENDING for further human review",
        memory=True,
        llm=st.session_state["crewllm"],
        #max_rpm=6,
        respect_context_window=True,
        #cache=True,
        verbose=True
    )
    review_task = Task(
        description="Review the following data from the document: {extracted_text} and extract key details such as the document type, claim type, service information, patient details, and other pertinent information. Categorize the claim type (e.g., medical, insurance, address change, etc..) and extract the most important details to assist in making an informed decision.",
        expected_output="The task will summarize the claim's content and provide context for decision-making. A structured summary of the claim, including the type of document, type of claim, and other key information like patient details, service date, etc after reviewing the following data: {extracted_text}",
        agent=review_agent
    )
    claim_task = Task(
        description="Make a decision based on your analysis of the data: {extracted_text}; and using your deep understanding of health insurance business process rules. Decide if the claim is APPROVED, PARTIALLY APPROVED, INELIGIBLE, or PENDING by referencing the business process rules only. Don't exaggerate. Documents which are NOT HEALTH INSURANCE RELATED are automatically INELIGIBLE. Return whether the claim is APPROVED, PARTIALLY APPROVED, INELIGIBLE, or PENDING, along with reasoning based on the provided input and business process rules. Keep the reasoning clear and to the point.",
        expected_output="A decision: APPROVED, PARTIALLY APPROVED, INELIGIBLE, or PENDING along with an explanation referring to the analyzed data.",
        output_json=DocumentReviewOutput,
        agent=claim_agent,
        context=[review_task]
    )
    approval_crew = Crew(
        agents=[review_agent, claim_agent],
        tasks=[review_task, claim_task],
        process=Process.sequential,
        verbose=True
    )
    return approval_crew

#############################
# Processing Functions for OCR and Vision
#############################
def process_ocr_pdf(file_path, s3_uri, from_s3):
    try:
        if from_s3:
            st.info("Processing PDF from S3 using Textract OCR...")
            textract_client = boto3.client("textract", region_name="us-east-2")
            loader = AmazonTextractPDFLoader(s3_uri, client=textract_client)
        else:
            loader = AmazonTextractPDFLoader(file_path)
        documents = loader.load()
        ocr_text = "\n\n".join([doc.page_content for doc in documents])
        llm = init_chat_model(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier",
            model_provider="together",
            api_key=together_api_key
        )
        chain = load_qa_chain(llm=llm, chain_type="map_reduce")
        qa_result = chain.run(input_documents=documents, question=query)
        return {"Method": "Textract OCR + QA Chain (PDF)", "Output": qa_result, "OCR Text": ocr_text}
    except Exception as e:
        return {"Method": "Textract OCR + QA Chain (PDF)", "Output": f"Error: {e}", "OCR Text": ""}

def process_vision_pdf(file_data):
    try:
        doc = fitz.open(stream=file_data, filetype="pdf")
        vision_outputs = []
        client = Together(api_key=together_api_key)
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            page_bytes = buffer.getvalue()
            page_encoded = base64.b64encode(page_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{page_encoded}"
            response = client.chat.completions.create(
                model=together_model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": f"{query} (Response for page {idx+1})"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]}
                ]
            )
            vision_outputs.append(f"Page {idx+1}: {response.choices[0].message.content}")
        vision_output_str = "\n".join(vision_outputs)
        return {"Method": "LLM Vision (PDF Pages)", "Output": vision_output_str}
    except Exception as e:
        return {"Method": "LLM Vision (PDF Pages)", "Output": f"Error: {e}"}

def process_ocr_image(file_data):
    try:
        textract_client = boto3.client("textract", region_name="us-east-2")
        response = textract_client.detect_document_text(Document={'Bytes': file_data})
        ocr_text = ""
        if "Blocks" in response:
            for block in response["Blocks"]:
                if block["BlockType"] == "LINE":
                    ocr_text += block["Text"] + "\n"
        doc = Document(page_content=ocr_text)
        llm = init_chat_model(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-classifier",
            model_provider="together",
            api_key=together_api_key
        )
        chain = load_qa_chain(llm=llm, chain_type="map_reduce")
        qa_result = chain.run(input_documents=[doc], question=query)
        return {"Method": "Textract OCR + QA Chain (Image)", "Output": qa_result, "OCR Text": ocr_text}
    except Exception as e:
        return {"Method": "Textract OCR + QA Chain (Image)", "Output": f"Error: {e}", "OCR Text": ""}

def process_vision_image(file_data):
    try:
        image = Image.open(io.BytesIO(file_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        encoded_img = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded_img}"
        client = Together(api_key=together_api_key)
        response = client.chat.completions.create(
            model=together_model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}
            ]
        )
        vision_output = response.choices[0].message.content
        return {"Method": "LLM Vision (Image)", "Output": vision_output}
    except Exception as e:
        return {"Method": "LLM Vision (Image)", "Output": f"Error: {e}"}

#############################
# Main Processing Block for Documents
#############################
if st.button("Process Document"):
    if not together_api_key:
        st.error("Please enter your Together AI API Key.")
    elif not query:
        st.error("Please enter a query or prompt.")
    elif not uploaded_file and not s3_bucket:
        st.error("Please either upload a file or enter an S3 bucket name.")
    else:
        with st.spinner("Processing documents concurrently..."):
            results = []  # To hold results for each file
            file_list = []
            if uploaded_file:
                file_data, file_name = get_file_data()
                file_list.append((file_data, file_name, None))
            else:
                st.info("Connecting to S3 bucket...")
                files_in_bucket = list_s3_files(s3_bucket, s3_prefix)
                st.write("Files found in bucket:", files_in_bucket)
                if not files_in_bucket:
                    st.error("No files found in the bucket with the specified prefix.")
                    st.stop()
                for key in files_in_bucket:
                    file_data, file_name = get_s3_file(s3_bucket, key)
                    s3_uri_for_file = f"s3://{s3_bucket}/{key}"
                    if file_data:
                        file_list.append((file_data, file_name, s3_uri_for_file))
            
            progress_bar = st.progress(0)
            status_log = st.empty()
            status_messages = []
            num_files = len(file_list)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_file = {}
                for tup in file_list:
                    file_data, file_name, s3_uri_for_file = tup
                    from_s3 = s3_uri_for_file is not None
                    if file_name.endswith(".pdf"):
                        st.info(f"Queuing PDF for processing: {file_name}")
                        if from_s3:
                            future = executor.submit(process_ocr_pdf, None, s3_uri_for_file, True)
                        else:
                            st.error(f"Local PDF file '{file_name}' cannot be processed due to Textract requirements. Please provide PDFs via S3.")
                            continue
                        future_to_file[future] = (file_name, "PDF OCR")
                    elif file_name.endswith((".png", ".jpg", ".jpeg")):
                        st.info(f"Queuing image for processing: {file_name}")
                        future = executor.submit(process_vision_image, file_data)
                        future_to_file[future] = (file_name, "Image Vision")
                    else:
                        st.warning(f"Unsupported file type: {file_name}")
                completed = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    file_name, method = future_to_file[future]
                    try:
                        res = future.result()
                        res["file_name"] = file_name
                        results.append(res)
                        status_messages.append(f"{file_name} processed using {method}.")
                    except Exception as exc:
                        results.append({"file_name": file_name, "Method": method, "Output": f"Error: {exc}"})
                        status_messages.append(f"{file_name} failed: {exc}")
                    completed += 1
                    progress_bar.progress(completed / num_files)
                    status_log.text("\n".join(status_messages))
            
            final_data = []
            for res in results:
                raw_text = res.get("OCR Text") if res.get("OCR Text") else res.get("Output", "")
                parsed_data = parse_claim_data(raw_text, res.get("Method", ""), res.get("file_name", ""))
                final_data.append(parsed_data)
            
            st.session_state.final_data = final_data
            pass
            
        st.success("Document processing complete!")

# Main UI: Aggregated & Individual Outputs and CrewAI Review
if "final_data" in st.session_state:
    aggregated_json = json.dumps(st.session_state.final_data, indent=2)
    st.subheader("Aggregated Extracted Claim Data (JSON)")
    st.text_area("All Documents Output", value=aggregated_json, height=400)
    st.download_button("Download All JSON", aggregated_json, file_name="extracted_claim_data.json")
    
    # Process CrewAI reviews if not already processed.
    if "review_results" not in st.session_state:
        st.info("Processing CrewAI reviews for all documents...")
        inputs_array = [
            {"extracted_text": entry["raw_output"], "id": entry["file_name"]}
            for entry in st.session_state.final_data
        ]
        review_crew = review_document_with_crewai()  # Builds the CrewAI crew
        review_results_array = review_crew.kickoff_for_each(inputs=inputs_array)
        review_results = {inp["id"]: result for inp, result in zip(inputs_array, review_results_array)}
        st.session_state.review_results = review_results

    # Initialize an ID counter in session_state if not already set
    if "current_id" not in st.session_state:
        st.session_state["current_id"] = 0
    if "claim" not in st.session_state:
        st.session_state["claim"] = 0

    # Insert each CrewAI review JSON (raw output) into Supabase.
    if st.button("Insert Review JSONs into Supabase"):
        for entry in st.session_state.final_data:
            file_id = entry["file_name"]
            if file_id in st.session_state.review_results:
                review_raw = st.session_state.review_results[file_id].raw
                try:
                    review_data = json.loads(review_raw)
                except Exception as e:
                    st.error(f"Error parsing review JSON for {file_id}: {e}")
                    continue

                # Build record with proper type conversion:
                try:
                    # Increment the counter for each record and set it as the id.
                    st.session_state["current_id"] += 1
                    st.session_state["claim"] += 1
                    
                    record = {
                        "id": st.session_state["current_id"],
                        "description": file_id,
                        "vendor": review_data.get("vendor"),
                        "service_type": review_data.get("service_type"),
                        "claim_id": int(review_data.get("claim_id")) if review_data.get("claim_id") is not None else int(st.session_state["claim"]),
                        "claim_amt": float(review_data.get("claim_amt")) if review_data.get("claim_amt") is not None else None,
                        "ai_decision": review_data.get("decision"),
                        "ai_reason": review_data.get("explanation"),
                        "raw_data": review_data  # Assuming your column is a JSON type; otherwise use json.dumps(review_data)
                    }
                except Exception as e:
                    st.error(f"Error building record for {file_id}: {e}")
                    continue

                try:
                    response = supabase.table("claim_data").insert(record).execute()
                    st.write(f"Inserted record for {file_id}: {response}")
                except Exception as e:
                    st.error(f"Error inserting record for {file_id}: {e}")
        st.success("All records inserted into Supabase!")


    
    # Sidebar: Show aggregated review decisions using the structured JSON output.
    st.sidebar.markdown("### CrewAI Review Decisions")
    file_options = [entry["file_name"] for entry in st.session_state.final_data]
    for fn in file_options:
        if fn in st.session_state.review_results:
            review_raw = st.session_state.review_results[fn].raw  # Expected to be a JSON string
            try:
                review_json = json.loads(review_raw)
                vendor = review_json.get("vendor", "N/A")
                service_type = review_json.get("service_type", "N/A")
                claim_amt = review_json.get("claim_amt", "N/A")
                decision = review_json.get("decision", "N/A")
                summary = f"Vendor: {vendor} | Service: {service_type} | Amt: {claim_amt} | Decision: {decision}"
            except Exception as e:
                summary = f"Error parsing review JSON: {e}"
            st.sidebar.markdown(f"**{fn}**: {summary}")
    
    # Sidebar: Select an individual document.
    selected_file = st.sidebar.selectbox("Select a Document", file_options)
    selected_entry = next((entry for entry in st.session_state.final_data if entry["file_name"] == selected_file), None)
    if selected_entry:
        selected_json = json.dumps(selected_entry, indent=2)
        with st.expander("View Individual Document JSON"):
            st.text_area("Document JSON", value=selected_json, height=400, key=f"individual_json_{selected_file}_{uuid.uuid4()}")
            st.download_button("Download Document JSON", selected_json, file_name=f"{selected_file}_extracted.json")
        
        # Display the CrewAI review result for the selected document.
        if selected_file in st.session_state.review_results:
            st.markdown("### Document Review for " + selected_file)
            review_result = st.session_state.review_results[selected_file]
            st.markdown("#### Document Review Decision")
            try:
                review_dict = json.loads(review_result.raw)
                st.json(review_dict)
            except Exception as e:
                st.write("Error parsing JSON:", e)
                st.write(review_result.raw)
            st.download_button("Download Review JSON", review_result.raw, file_name=f"{selected_file}_review.json")
            st.markdown("#### Token Usage for Review")
            st.write(review_result.token_usage)
            st.markdown("#### Task Outputs for Review")
            st.write_stream(review_result.tasks_output)
    
    st.success("Processing complete!")

