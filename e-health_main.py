import os
import json
import traceback
from typing import Dict, Any, List, Literal

import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_PATH = os.path.join(BASE_DIR, "prompt", "prompt.md")
TEMP_DIR = os.path.join(BASE_DIR, "temp")


# --------------------------------------------------
# STATE
# --------------------------------------------------
class HealthGraphState(TypedDict, total=False):
    username: str
    password: str
    uploaded_file_path: str
    prompt_file: str

    is_authenticated: bool
    auth_error: str

    saved_pdf_path: str
    pages_count: int
    full_text: str
    extraction_method: str
    fallback_used: bool

    extracted_json_text: str
    data: Dict[str, Any]
    abnormal_tests: List[Dict[str, Any]]
    recommendation_data: NotRequired[Dict[str, Any]]
    pdf_output_path: str
    test_count: int

    error: str
    status: str


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def clean_json_text(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def get_abnormal_tests(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        test
        for test in data.get("laboratory_test_results", [])
        if str(test.get("status", "")).lower() != "normal"
    ]


def extract_pdfplumber_text_and_tables(pdf_path: str) -> tuple[str, int]:
    extracted_pages = []
    table_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                extracted_pages.append(f"--- Page {page_num} Text ---\n{text.strip()}")

            tables = page.extract_tables() or []
            for table_idx, table in enumerate(tables, start=1):
                if not table:
                    continue

                cleaned_rows = []
                for row in table:
                    if not row:
                        continue
                    cleaned_row = [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    if any(cell for cell in cleaned_row):
                        cleaned_rows.append(cleaned_row)

                if cleaned_rows:
                    block_lines = [f"--- Page {page_num} Table {table_idx} ---"]
                    for row in cleaned_rows:
                        block_lines.append(" | ".join(row))
                    table_blocks.append("\n".join(block_lines))

        combined_parts = extracted_pages[:]
        if table_blocks:
            combined_parts.append("EXTRACTED TABLES:\n" + "\n\n".join(table_blocks))

        return "\n\n".join(combined_parts).strip(), len(pdf.pages)


def generate_health_pdf(data: Dict[str, Any], pdf_path: str):
    doc = SimpleDocTemplate(
        pdf_path,
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )

    styles = getSampleStyleSheet()
    elements = []

    title_table = Table(
        [[
            Paragraph(
                "<font color='white'><b>E-Health Insight Report</b></font>",
                styles["Title"]
            )
        ]],
        colWidths=[540]
    )

    title_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#006D77")),
            ("BOX", (0, 0), (-1, -1), 2, colors.HexColor("#0D47A1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ])
    )

    elements.append(title_table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Patient Information</b>", styles["Heading2"]))

    patient_info = f"""
    <b>Name:</b> {data.get('patient_name', '')}<br/>
    <b>Age:</b> {data.get('age', '')}<br/>
    <b>Gender:</b> {data.get('gender', '')}
    """
    elements.append(Paragraph(patient_info, styles["BodyText"]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>Lab Information</b>", styles["Heading2"]))

    lab_info = f"""
    <b>Provider:</b> {data.get('lab_report_provider', '')}<br/>
    <b>Lab ID:</b> {data.get('lab_id', '')}<br/>
    <b>Center:</b> {data.get('center', '')}<br/>
    <b>Collection:</b> {data.get('collection_date_time', '')}<br/>
    <b>Reporting:</b> {data.get('reporting_date_time', '')}<br/>
    <b>Ref Doctor:</b> {data.get('ref_doctor', '')}
    """
    elements.append(Paragraph(lab_info, styles["BodyText"]))
    elements.append(Spacer(1, 20))

    tests = data.get("laboratory_test_results", [])
    total_tests = len(tests)
    abnormal_count = len([
        t for t in tests
        if str(t.get("status", "")).lower() != "normal"
    ])
    normal_count = total_tests - abnormal_count

    elements.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    elements.append(
        Paragraph(
            f"""
            Total Tests: <b>{total_tests}</b><br/>
            Normal: <b>{normal_count}</b><br/>
            Abnormal: <font color="red"><b>{abnormal_count}</b></font>
            """,
            styles["BodyText"]
        )
    )
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Laboratory Results</b>", styles["Heading2"]))

    table_data = [[
        Paragraph("<b>Test Name</b>", styles["BodyText"]),
        Paragraph("<b>Value</b>", styles["BodyText"]),
        Paragraph("<b>Unit</b>", styles["BodyText"]),
        Paragraph("<b>Reference</b>", styles["BodyText"]),
        Paragraph("<b>Status</b>", styles["BodyText"])
    ]]

    for test in tests:
        status = str(test.get("status", ""))

        if status.lower() != "normal":
            status_para = Paragraph(
                f'<font color="red"><b>{status}</b></font>',
                styles["BodyText"]
            )
        else:
            status_para = Paragraph(status, styles["BodyText"])

        table_data.append([
            Paragraph(str(test.get("test_name", "")), styles["BodyText"]),
            Paragraph(str(test.get("observed_value", "")), styles["BodyText"]),
            Paragraph(str(test.get("unit", "")), styles["BodyText"]),
            Paragraph(str(test.get("reference_range", "")), styles["BodyText"]),
            status_para
        ])

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[220, 55, 60, 95, 60]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3F2FD")),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        ])
    )

    elements.append(table)
    elements.append(PageBreak())

    elements.append(Paragraph("AI Health Recommendations:", styles["Heading1"]))
    elements.append(Spacer(1, 10))

    recommendation_number = 1

    for test in tests:
        if str(test.get("status", "")).lower() == "normal":
            continue

        card_data = [[
            Paragraph(
                f"""
                <font color="#800000"><b>🔴 Finding #{recommendation_number}</b></font><br/><br/>
                <font color="red"><b>{test.get('test_name', '')}</b></font><br/><br/>
                <b>Status:</b> <font color="red"><b>{test.get('status', '')}</b></font><br/><br/>
                <b>Analysis:</b><br/>{test.get('analysis', 'Not Available')}<br/><br/>
                <b>Possible Causes:</b><br/>{test.get('possible_causes', 'Not Available')}<br/><br/>
                <b>Diet Recommendation:</b><br/>{test.get('diet_recommendation', 'Not Available')}<br/><br/>
                <b>Lifestyle Recommendation:</b><br/>{test.get('lifestyle_recommendation', 'Not Available')}<br/><br/>
                <b>Doctor Consultation:</b><br/>{test.get('doctor_consultation', 'Not Available')}
                """,
                styles["BodyText"]
            )
        ]]

        card = Table(card_data, colWidths=[520])
        card.setStyle(
            TableStyle([
                ("BOX", (0, 0), (-1, -1), 2, colors.red),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF5F5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ])
        )

        elements.append(card)
        elements.append(Spacer(1, 15))
        recommendation_number += 1

    doc.build(elements)


# --------------------------------------------------
# NODES
# --------------------------------------------------
def auth_node(state: HealthGraphState) -> Dict[str, Any]:
    if state["username"] == "admin" and state["password"] == "Delhi@12345":
        return {
            "is_authenticated": True,
            "status": "authenticated"
        }

    return {
        "is_authenticated": False,
        "auth_error": "Invalid credentials",
        "error": "Authentication failed"
    }


def pypdf_extraction_node(state: HealthGraphState) -> Dict[str, Any]:
    pdf_path = state["uploaded_file_path"]
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    full_text = "\n".join(
        (page.page_content or "").strip()
        for page in pages
    ).strip()

    return {
        "saved_pdf_path": pdf_path,
        "pages_count": len(pages),
        "full_text": full_text,
        "extraction_method": "PyPDFLoader",
        "fallback_used": False,
        "status": "pypdf_extraction_done"
    }


def pdfplumber_extraction_node(state: HealthGraphState) -> Dict[str, Any]:
    pdf_path = state["uploaded_file_path"]
    combined_text, page_count = extract_pdfplumber_text_and_tables(pdf_path)

    return {
        "saved_pdf_path": pdf_path,
        "pages_count": page_count,
        "full_text": combined_text,
        "extraction_method": "pdfplumber+tables",
        "fallback_used": True,
        "status": "pdfplumber_extraction_done"
    }


def llm_extraction_node(state: HealthGraphState) -> Dict[str, Any]:
    with open(state["prompt_file"], "r", encoding="utf-8") as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["context"],
        template=template
    )

    final_prompt = prompt.format(context=state["full_text"])

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke(final_prompt)
    cleaned = clean_json_text(response.content)
    data = json.loads(cleaned)

    tests = data.get("laboratory_test_results", [])
    if not isinstance(tests, list):
        tests = []

    return {
        "extracted_json_text": cleaned,
        "data": data,
        "test_count": len(tests),
        "status": "lab_json_extracted"
    }


def llm_extraction_from_pypdf_node(state: HealthGraphState) -> Dict[str, Any]:
    return llm_extraction_node(state)


def llm_extraction_from_pdfplumber_node(state: HealthGraphState) -> Dict[str, Any]:
    return llm_extraction_node(state)


def abnormal_router_node(state: HealthGraphState) -> Dict[str, Any]:
    abnormal_tests = get_abnormal_tests(state.get("data", {}))

    if not abnormal_tests:
        return {
            "abnormal_tests": [],
            "recommendation_data": {
                "overall_health_status": "Healthy",
                "recommendations": []
            },
            "status": "no_abnormal_tests"
        }

    return {
        "abnormal_tests": abnormal_tests,
        "status": "abnormal_tests_identified"
    }


def recommendation_agent_node(state: HealthGraphState) -> Dict[str, Any]:
    abnormal_tests = state.get("abnormal_tests", [])

    if not abnormal_tests:
        return {
            "recommendation_data": {
                "overall_health_status": "Healthy",
                "recommendations": []
            },
            "status": "no_abnormal_tests"
        }

    recommendation_prompt = f"""
You are an expert healthcare assistant.

Analyze ONLY the abnormal laboratory findings.

{json.dumps(abnormal_tests, indent=2)}

For each abnormal test provide:

1. analysis
2. possible_causes
3. diet_recommendation
4. lifestyle_recommendation
5. doctor_consultation

Return ONLY valid JSON.

{{
    "overall_health_status":"",
    "recommendations":[
        {{
            "test_name":"",
            "analysis":"",
            "possible_causes":"",
            "diet_recommendation":"",
            "lifestyle_recommendation":"",
            "doctor_consultation":""
        }}
    ]
}}
"""

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.2
    )

    response = llm.invoke(recommendation_prompt)
    cleaned = clean_json_text(response.content)
    recommendation_data = json.loads(cleaned)

    return {
        "recommendation_data": recommendation_data,
        "status": "recommendations_generated"
    }


def merge_results_node(state: HealthGraphState) -> Dict[str, Any]:
    data = state.get("data", {})

    recommendation_data = state.get(
        "recommendation_data",
        {
            "overall_health_status": "Healthy",
            "recommendations": []
        }
    )

    recommendation_map = {
        rec.get("test_name"): rec
        for rec in recommendation_data.get("recommendations", [])
        if rec.get("test_name")
    }

    for test in data.get("laboratory_test_results", []):
        test_name = test.get("test_name")
        if test_name in recommendation_map:
            rec = recommendation_map[test_name]
            test["analysis"] = rec.get("analysis", "")
            test["possible_causes"] = rec.get("possible_causes", "")
            test["diet_recommendation"] = rec.get("diet_recommendation", "")
            test["lifestyle_recommendation"] = rec.get("lifestyle_recommendation", "")
            test["doctor_consultation"] = rec.get("doctor_consultation", "")

    if "overall_health_status" not in data:
        data["overall_health_status"] = recommendation_data.get(
            "overall_health_status",
            "Healthy"
        )

    return {
        "data": data,
        "status": "results_merged"
    }


def pdf_report_node(state: HealthGraphState) -> Dict[str, Any]:
    os.makedirs(TEMP_DIR, exist_ok=True)
    patient_name = state.get("data", {}).get("patient_name", "patient")
    safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in patient_name).strip()
    safe_name = safe_name.replace(" ", "_") or "patient"

    pdf_output_path = os.path.join(TEMP_DIR, f"{safe_name}_Health_Report.pdf")
    generate_health_pdf(state["data"], pdf_output_path)

    return {
        "pdf_output_path": pdf_output_path,
        "status": "pdf_generated"
    }


# --------------------------------------------------
# ROUTERS
# --------------------------------------------------
def auth_route(state: HealthGraphState) -> Literal["pypdf_extraction", "__end__"]:
    return "pypdf_extraction" if state.get("is_authenticated") else END


def test_count_route_after_pypdf(state: HealthGraphState) -> Literal["abnormal_router", "pdfplumber_extraction"]:
    tests = state.get("data", {}).get("laboratory_test_results", [])
    if not isinstance(tests, list) or len(tests) == 0:
        return "pdfplumber_extraction"
    return "abnormal_router"


def test_count_route_after_pdfplumber(state: HealthGraphState) -> Literal["abnormal_router", "merge_results"]:
    tests = state.get("data", {}).get("laboratory_test_results", [])
    if not isinstance(tests, list) or len(tests) == 0:
        data = state.get("data", {})
        if "overall_health_status" not in data:
            data["overall_health_status"] = "Parsing Failed or No Tests Found"
        return "merge_results"
    return "abnormal_router"


def abnormal_route(state: HealthGraphState) -> Literal["recommendation_agent", "merge_results"]:
    return "recommendation_agent" if state.get("abnormal_tests") else "merge_results"


# --------------------------------------------------
# GRAPH
# --------------------------------------------------
@st.cache_resource
def build_graph():
    builder = StateGraph(HealthGraphState)

    builder.add_node("auth", auth_node)
    builder.add_node("pypdf_extraction", pypdf_extraction_node)
    builder.add_node("pdfplumber_extraction", pdfplumber_extraction_node)
    builder.add_node("llm_extraction_from_pypdf", llm_extraction_from_pypdf_node)
    builder.add_node("llm_extraction_from_pdfplumber", llm_extraction_from_pdfplumber_node)
    builder.add_node("abnormal_router", abnormal_router_node)
    builder.add_node("recommendation_agent", recommendation_agent_node)
    builder.add_node("merge_results", merge_results_node)
    builder.add_node("pdf_report", pdf_report_node)

    builder.add_edge(START, "auth")

    builder.add_conditional_edges(
        "auth",
        auth_route,
        {
            "pypdf_extraction": "pypdf_extraction",
            END: END
        }
    )

    builder.add_edge("pypdf_extraction", "llm_extraction_from_pypdf")

    builder.add_conditional_edges(
        "llm_extraction_from_pypdf",
        test_count_route_after_pypdf,
        {
            "abnormal_router": "abnormal_router",
            "pdfplumber_extraction": "pdfplumber_extraction"
        }
    )

    builder.add_edge("pdfplumber_extraction", "llm_extraction_from_pdfplumber")

    builder.add_conditional_edges(
        "llm_extraction_from_pdfplumber",
        test_count_route_after_pdfplumber,
        {
            "abnormal_router": "abnormal_router",
            "merge_results": "merge_results"
        }
    )

    builder.add_conditional_edges(
        "abnormal_router",
        abnormal_route,
        {
            "recommendation_agent": "recommendation_agent",
            "merge_results": "merge_results"
        }
    )

    builder.add_edge("recommendation_agent", "merge_results")
    builder.add_edge("merge_results", "pdf_report")
    builder.add_edge("pdf_report", END)

    return builder.compile()


graph = build_graph()


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="Yours E-Health Insider",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e6f7ff, #f9fcff);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #006D77, #83C5BE);
    }

    section[data-testid="stSidebar"] * {
        color: black !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 15px;
    }

    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border: 2px dashed #83C5BE;
    }

    div[data-testid="stTextInput"] input {
        background-color: #f8fbff;
        border: 1px solid #bcdff1;
        border-radius: 10px;
        color: #003049;
    }

    div[data-testid="stTextInput"] label {
        color: #003049 !important;
        font-weight: 600;
    }

    h1 {
        color: #003049;
        text-align: center;
    }

    h4 {
        text-align: center;
        color: #5c677d;
    }

    .stButton > button {
        background-color: #006D77;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.2rem;
    }

    .stButton > button:hover {
        background-color: #00565e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(to right, #006D77, #83C5BE);
        color: white;
        padding: 10px 0;
        z-index: 999999;
        overflow: hidden;
        border-top: 2px solid #ffffff33;
    }

    .footer-text {
        display: inline-block;
        white-space: nowrap;
        padding-left: 100%;
        animation: marquee 15s linear infinite;
        font-size: 16px;
        font-weight: 600;
    }

    @keyframes marquee {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .block-container {
        padding-bottom: 70px;
    }
    </style>

    <div class="custom-footer">
        <div class="footer-text">
            📌 Prepared by Rohit Sotra | AI-Powered E-Health Insight Report | Upload your lab report and get instant health insights
        </div>
    </div>
""", unsafe_allow_html=True)

st.title("📑 Get instant Health Report for your Lab Test!!")
#st.markdown(
#    "<h4>Prepared by Rohit Sotra</h4>",
#    unsafe_allow_html=True
#)

st.sidebar.header("Login")
username = st.sidebar.text_input("Enter Username:")
user_password = st.sidebar.text_input("Enter Password:", type="password")

uploaded_file = st.file_uploader(
    "Upload Lab Report",
    type=["pdf"]
)

#prompt_file = st.text_input(
#    "Prompt file path",
#    value=r"F:\GEN_AI\Case_Study\EHealth_insight\prompt\prompt.md"
#)

#prompt_file = r"F:\GEN_AI\Case_Study\EHealth_insight\prompt\prompt.md"
BASE_DIR = Path(__file__).resolve().parent
prompt_file = BASE_DIR / "prompt" / "prompt.md"

if uploaded_file is not None:
    st.info(f"Uploaded file: {uploaded_file.name}")

if st.button("Process Lab Report"):
    if uploaded_file is None:
        st.error("Please upload a PDF file.")
    elif not username or not user_password:
        st.error("Please enter username and password.")
    elif not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing in environment variables.")
    elif not os.path.exists(prompt_file):
        st.error(f"Prompt file path does not exist: {prompt_file}")
        st.write("Resolved path:", os.path.abspath(prompt_file))
    else:
        try:
            with st.spinner("Processing Lab Report..."):
                os.makedirs(TEMP_DIR, exist_ok=True)
                saved_pdf_path = os.path.join(TEMP_DIR, uploaded_file.name)

                with open(saved_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                result = graph.invoke({
                    "username": username,
                    "password": user_password,
                    "uploaded_file_path": saved_pdf_path,
                    "prompt_file": str(prompt_file)
                })

            if result.get("error"):
                st.error(result["error"])
            elif not result.get("is_authenticated", False):
                st.error(result.get("auth_error", "Authentication failed"))
            else:
                st.success("Lab Report Processed Successfully")

                st.write("**Extraction Method Used:**", result.get("extraction_method", "Unknown"))
                st.write("**Fallback Used:**", result.get("fallback_used", False))
                st.write("**Pages Loaded:**", result.get("pages_count", 0))
                st.write("**Tests Count:**", len(result.get("data", {}).get("laboratory_test_results", [])))

                data = result.get("data", {})
                st.subheader("Extracted JSON")
                st.json(data)

                if len(data.get("laboratory_test_results", [])) == 0:
                    st.warning("No laboratory tests were parsed from the report. Fallback was attempted.")

                st.subheader("🩺 AI Health Insights")
                st.success(
                    f"Overall Health Status: {data.get('overall_health_status', 'Not Available')}"
                )

                for test in data.get("laboratory_test_results", []):
                    if str(test.get("status", "")).lower() != "normal":
                        with st.expander(f"🔴 {test.get('test_name', '')} ({test.get('status', '')})"):
                            st.write(f"**Analysis:** {test.get('analysis', '')}")
                            st.write(f"**Possible Causes:** {test.get('possible_causes', '')}")
                            st.write(f"**Diet Recommendation:** {test.get('diet_recommendation', '')}")
                            st.write(f"**Lifestyle Recommendation:** {test.get('lifestyle_recommendation', '')}")
                            st.write(f"**Doctor Consultation:** {test.get('doctor_consultation', '')}")

                pdf_output_path = result.get("pdf_output_path")
                if pdf_output_path and os.path.exists(pdf_output_path):
                    with open(pdf_output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Health Report PDF",
                            data=f,
                            file_name=os.path.basename(pdf_output_path),
                            mime="application/pdf"
                        )

        except Exception as e:
            st.error(f"Application Error: {str(e)}")
            st.subheader("Full Traceback")
            st.code(traceback.format_exc())
