import streamlit as st
import json
import logging
from typing import Dict, List, Any, TypedDict
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ResumeData(BaseModel):
    """Structured resume data"""
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address if found", default="Not specified")
    phone: str = Field(description="Phone number if found", default="Not specified")
    summary: str = Field(description="Professional summary or objective")
    education: List[Dict[str, str]] = Field(description="List of education entries with degree, institution, year")
    experience: List[Dict[str, str]] = Field(description="List of work experience with title, company, duration, responsibilities")
    skills: List[str] = Field(description="List of technical and soft skills")
    certifications: List[str] = Field(description="List of certifications if any", default_factory=list)
    projects: List[Dict[str, str]] = Field(description="List of projects with name, description, technologies", default_factory=list)
    languages: List[str] = Field(description="Languages spoken if mentioned", default_factory=list)

class JobDescriptionData(BaseModel):
    """Structured job description data"""
    job_title: str = Field(description="Job title")
    company: str = Field(description="Company name if mentioned", default="Not specified")
    location: str = Field(description="Job location", default="Not specified")
    summary: str = Field(description="Job summary or overview")
    responsibilities: List[str] = Field(description="List of job responsibilities")
    required_skills: List[str] = Field(description="Required technical skills")
    preferred_skills: List[str] = Field(description="Preferred or nice-to-have skills", default_factory=list)
    experience_level: str = Field(description="Required experience level (entry, mid, senior, etc.)")
    education_requirements: str = Field(description="Education requirements")
    soft_skills: List[str] = Field(description="Required soft skills", default_factory=list)

class MatchAnalysis(BaseModel):
    """Comprehensive match analysis"""
    overall_score: float = Field(description="Overall match score from 0.0 to 1.0")
    skill_match_score: float = Field(description="Skills match score from 0.0 to 1.0")
    experience_match_score: float = Field(description="Experience match score from 0.0 to 1.0")
    matched_skills: List[str] = Field(description="Skills that match between resume and JD")
    missing_skills: List[str] = Field(description="Skills required by JD but missing in resume")
    experience_gaps: List[str] = Field(description="Experience gaps or mismatches")
    strengths: List[str] = Field(description="Candidate's strengths for this role")
    weaknesses: List[str] = Field(description="Areas where candidate falls short")
    recommendations: List[str] = Field(description="Specific recommendations for improvement")
    risk_factors: List[str] = Field(description="Potential risk factors")
    seniority_assessment: str = Field(description="Assessment of candidate's seniority level")

class ResumeEnhancement(BaseModel):
    """Resume enhancement suggestions"""
    summary_improvements: List[str] = Field(description="Suggestions to improve the professional summary")
    experience_enhancements: List[Dict[str, str]] = Field(description="Specific improvements for each work experience")
    skill_additions: List[str] = Field(description="Skills to add or emphasize")
    formatting_suggestions: List[str] = Field(description="Formatting and structure improvements")
    keyword_optimization: List[str] = Field(description="Keywords to include for ATS optimization")
    overall_recommendations: List[str] = Field(description="Overall recommendations for resume improvement")

class CoverLetter(BaseModel):
    """Generated cover letter"""
    opening: str = Field(description="Opening paragraph")
    body: str = Field(description="Main body paragraphs")
    closing: str = Field(description="Closing paragraph")
    full_letter: str = Field(description="Complete cover letter")


class AgentState(TypedDict):
    """State for the LangGraph workflow"""
    resume_text: str
    jd_text: str
    parsed_resume: ResumeData
    parsed_jd: JobDescriptionData
    match_analysis: MatchAnalysis
    resume_enhancement: ResumeEnhancement
    cover_letter: CoverLetter
    current_step: str
    errors: List[str]
    workflow_type: str  

# ==================== LLM SETUP ====================

def get_llm():
    """Initialize OpenAI LLM"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise Exception("Please set your OPENAI_API_KEY in the .env file")
        
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=4000,
            api_key=api_key
        )
    except Exception as e:
        logger.error(f"OpenAI setup failed: {e}")
        raise Exception(f"OpenAI configuration error: {str(e)}. Please check your .env file.")


class ResumeParsingAgent:
    """Resume Parser Agent - extracts structured data from resumes"""
    
    def __init__(self, llm):
        self.llm = llm
        self.role = "Resume Parser"
        self.backstory = "I extract structured data from resumes with high accuracy."
        self.goal = "Parse resumes to extract skills, experience, education, certifications, and career gaps."
    
    def execute_task(self, state: AgentState) -> AgentState:
        """Parse and structure resume data"""
        try:
            parser = JsonOutputParser(pydantic_object=ResumeData)
            format_instructions = parser.get_format_instructions()
            
            system_prompt = f"""You are a {self.role}. {self.backstory}
Goal: {self.goal}

IMPORTANT GUIDELINES:
1. Extract ALL available information accurately
2. If information is not present, use "Not specified" or empty lists
3. For experience, include role, company, duration, and key responsibilities
4. For skills, separate technical and soft skills appropriately
5. Maintain the original meaning and context
6. Be thorough but concise in descriptions

{format_instructions}"""
            
            human_prompt = f"""Please parse the following resume and extract structured data:

{state['resume_text']}

Return the data in the exact JSON format specified."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            result = self.llm.invoke(messages)
            parsed_result = parser.parse(result.content)
            
            return {
                **state,
                "parsed_resume": parsed_result,
                "current_step": "resume_parsed"
            }
        except Exception as e:
            logger.error(f"Resume parsing failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Resume parsing failed: {str(e)}"],
                "current_step": "error"
            }

class JDUnderstandingAgent:
    """Job Description Understanding Agent - analyzes job descriptions"""
    
    def __init__(self, llm):
        self.llm = llm
        self.role = "JD Parser"
        self.backstory = "I analyze job descriptions to extract key requirements."
        self.goal = "Extract mandatory/optional skills, seniority, and soft skills from job descriptions."
    
    def execute_task(self, state: AgentState) -> AgentState:
        """Parse and structure job description data"""
        try:
            parser = JsonOutputParser(pydantic_object=JobDescriptionData)
            format_instructions = parser.get_format_instructions()
            
            system_prompt = f"""You are a {self.role}. {self.backstory}
Goal: {self.goal}

IMPORTANT GUIDELINES:
1. Extract ALL available information accurately
2. Distinguish between required and preferred skills
3. Identify experience level requirements
4. Extract both technical and soft skills
5. Maintain the original meaning and context
6. Be thorough but concise in descriptions

{format_instructions}"""
            
            human_prompt = f"""Please parse the following job description and extract structured data:

{state['jd_text']}

Return the data in the exact JSON format specified."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            result = self.llm.invoke(messages)
            parsed_result = parser.parse(result.content)
            
            return {
                **state,
                "parsed_jd": parsed_result,
                "current_step": "jd_parsed"
            }
        except Exception as e:
            logger.error(f"JD parsing failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"JD parsing failed: {str(e)}"],
                "current_step": "error"
            }

class MatchingAgent:
    """Candidate-Role Matcher Agent - computes match between resume and JD"""
    
    def __init__(self, llm):
        self.llm = llm
        self.role = "Candidate-Role Matcher"
        self.backstory = "I match candidates to roles based on skills and experience."
        self.goal = "Compute a match score between resume and job description."
    
    def execute_task(self, state: AgentState) -> AgentState:
        """Analyze match between resume and job description"""
        try:
            parser = JsonOutputParser(pydantic_object=MatchAnalysis)
            format_instructions = parser.get_format_instructions()
            
            # Handle both Pydantic model instances and dictionaries
            if hasattr(state["parsed_resume"], 'dict'):
                resume_data = state["parsed_resume"].dict()
            else:
                resume_data = state["parsed_resume"]
                
            if hasattr(state["parsed_jd"], 'dict'):
                jd_data = state["parsed_jd"].dict()
            else:
                jd_data = state["parsed_jd"]
            
            system_prompt = f"""You are a {self.role}. {self.backstory}
Goal: {self.goal}

ANALYSIS FRAMEWORK:
1. Skills Assessment: Compare technical and soft skills
2. Experience Evaluation: Assess experience level and relevance
3. Gap Analysis: Identify missing requirements
4. Risk Assessment: Identify potential concerns
5. Recommendations: Provide actionable advice

SCORING GUIDELINES:
- 0.9-1.0: Excellent match, highly recommended
- 0.7-0.8: Good match, recommended with minor concerns
- 0.5-0.6: Moderate match, consider with reservations
- 0.3-0.4: Poor match, not recommended
- 0.0-0.2: Very poor match, strongly not recommended

{format_instructions}"""
            
            human_prompt = f"""Please analyze the match between this candidate and job description:

RESUME DATA:
{json.dumps(resume_data, indent=2)}

JOB DESCRIPTION DATA:
{json.dumps(jd_data, indent=2)}

Provide a comprehensive match analysis in the exact JSON format specified."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            result = self.llm.invoke(messages)
            parsed_result = parser.parse(result.content)
            
            return {
                **state,
                "match_analysis": parsed_result,
                "current_step": "match_analyzed"
            }
        except Exception as e:
            logger.error(f"Match analysis failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Match analysis failed: {str(e)}"],
                "current_step": "error"
            }

class ResumeEnhancerAgent:
    """Resume Enhancer Agent - suggests improvements to resumes"""
    
    def __init__(self, llm):
        self.llm = llm
        self.role = "Resume Enhancer"
        self.backstory = "I suggest improvements to the resume to make it more relevant to the JD."
        self.goal = "Optimize resumes based on job descriptions."
    
    def execute_task(self, state: AgentState) -> AgentState:
        """Generate resume enhancement suggestions"""
        try:
            parser = JsonOutputParser(pydantic_object=ResumeEnhancement)
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""You are a {self.role}. {self.backstory}
                Goal: {self.goal}
                
                ENHANCEMENT GUIDELINES:
                1. Focus on ATS optimization with relevant keywords
                2. Suggest specific improvements for each section
                3. Address gaps identified in the match analysis
                4. Provide actionable, specific recommendations
                5. Consider industry standards and best practices
                6. Maintain professional tone and formatting"""),
                HumanMessage(content="""Please provide resume enhancement suggestions based on this analysis:

RESUME DATA:
{resume_data}

JOB DESCRIPTION DATA:
{jd_data}

MATCH ANALYSIS:
{match_analysis}

Provide specific enhancement suggestions in the exact JSON format specified by the parser.""")
            ])
            
            chain = prompt | self.llm | parser
            result = chain.invoke({
                "resume_data": json.dumps(state["parsed_resume"].dict(), indent=2),
                "jd_data": json.dumps(state["parsed_jd"].dict(), indent=2),
                "match_analysis": json.dumps(state["match_analysis"].dict(), indent=2)
            })
            
            return {
                **state,
                "resume_enhancement": result,
                "current_step": "enhancement_generated"
            }
        except Exception as e:
            logger.error(f"Resume enhancement failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Resume enhancement failed: {str(e)}"],
                "current_step": "error"
            }

class CoverLetterAgent:
    """Cover Letter Generator Agent - creates tailored cover letters"""
    
    def __init__(self, llm):
        self.llm = llm
        self.role = "Cover Letter Generator"
        self.backstory = "I write cover letters tailored to job descriptions and candidate profiles."
        self.goal = "Generate persuasive cover letters that address gaps and emphasize fit."
    
    def execute_task(self, state: AgentState) -> AgentState:
        """Generate a tailored cover letter"""
        try:
            parser = JsonOutputParser(pydantic_object=CoverLetter)
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""You are a {self.role}. {self.backstory}
                Goal: {self.goal}
                
                COVER LETTER GUIDELINES:
                1. Opening: Engaging hook that connects to the role
                2. Body: Specific examples and achievements that match job requirements
                3. Closing: Strong call to action and professional closing
                4. Address gaps identified in match analysis proactively
                5. Use specific keywords from the job description
                6. Maintain professional, confident tone
                7. Keep it concise (300-400 words total)"""),
                HumanMessage(content="""Please generate a tailored cover letter based on this analysis:

RESUME DATA:
{resume_data}

JOB DESCRIPTION DATA:
{jd_data}

MATCH ANALYSIS:
{match_analysis}

Generate a compelling cover letter in the exact JSON format specified by the parser.""")
            ])
            
            chain = prompt | self.llm | parser
            result = chain.invoke({
                "resume_data": json.dumps(state["parsed_resume"].dict(), indent=2),
                "jd_data": json.dumps(state["parsed_jd"].dict(), indent=2),
                "match_analysis": json.dumps(state["match_analysis"].dict(), indent=2)
            })
            
            return {
                **state,
                "cover_letter": result,
                "current_step": "cover_letter_generated"
            }
        except Exception as e:
            logger.error(f"Cover letter generation failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Cover letter generation failed: {str(e)}"],
                "current_step": "error"
            }

# ==================== LANGGRAPH WORKFLOW FUNCTIONS ====================

def resume_parser_node(state: AgentState) -> AgentState:
    """Resume parsing node for LangGraph"""
    llm = get_llm()
    agent = ResumeParsingAgent(llm)
    return agent.execute_task(state)

def jd_parser_node(state: AgentState) -> AgentState:
    """JD parsing node for LangGraph"""
    llm = get_llm()
    agent = JDUnderstandingAgent(llm)
    return agent.execute_task(state)

def matching_agent_node(state: AgentState) -> AgentState:
    """Matching agent node for LangGraph"""
    llm = get_llm()
    agent = MatchingAgent(llm)
    return agent.execute_task(state)

def resume_enhancer_node(state: AgentState) -> AgentState:
    """Resume enhancer node for LangGraph"""
    llm = get_llm()
    agent = ResumeEnhancerAgent(llm)
    return agent.execute_task(state)

def cover_letter_agent_node(state: AgentState) -> AgentState:
    """Cover letter agent node for LangGraph"""
    llm = get_llm()
    agent = CoverLetterAgent(llm)
    return agent.execute_task(state)



def create_matching_workflow() -> StateGraph:
    """Create workflow for matching analysis"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse_resume", resume_parser_node)
    workflow.add_node("parse_jd", jd_parser_node)
    workflow.add_node("match_analysis", matching_agent_node)
    
    workflow.set_entry_point("parse_resume")
    workflow.add_edge("parse_resume", "parse_jd")
    workflow.add_edge("parse_jd", "match_analysis")
    workflow.add_edge("match_analysis", END)
    
    return workflow.compile()

def create_enhancement_workflow() -> StateGraph:
    """Create workflow for resume enhancement"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse_resume", resume_parser_node)
    workflow.add_node("parse_jd", jd_parser_node)
    workflow.add_node("match_analysis", matching_agent_node)
    workflow.add_node("enhance_resume", resume_enhancer_node)
    
    workflow.set_entry_point("parse_resume")
    workflow.add_edge("parse_resume", "parse_jd")
    workflow.add_edge("parse_jd", "match_analysis")
    workflow.add_edge("match_analysis", "enhance_resume")
    workflow.add_edge("enhance_resume", END)
    
    return workflow.compile()

def create_cover_letter_workflow() -> StateGraph:
    """Create workflow for cover letter generation"""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse_resume", resume_parser_node)
    workflow.add_node("parse_jd", jd_parser_node)
    workflow.add_node("match_analysis", matching_agent_node)
    workflow.add_node("generate_cover_letter", cover_letter_agent_node)
    
    workflow.set_entry_point("parse_resume")
    workflow.add_edge("parse_resume", "parse_jd")
    workflow.add_edge("parse_jd", "match_analysis")
    workflow.add_edge("match_analysis", "generate_cover_letter")
    workflow.add_edge("generate_cover_letter", END)
    
    return workflow.compile()



def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def format_match_score(score: float) -> str:
    """Format match score with color coding"""
    if score >= 0.8:
        return f"🟢 {score:.1%} (Excellent Match)"
    elif score >= 0.6:
        return f"🟡 {score:.1%} (Good Match)"
    elif score >= 0.4:
        return f"🟠 {score:.1%} (Moderate Match)"
    else:
        return f"🔴 {score:.1%} (Poor Match)"




def main():
    st.set_page_config(page_title="🧠 Resume-JD Matcher", layout="wide")
    st.markdown("# 🧠 Resume-JD Matcher")
    st.markdown("AI-powered resume analysis using OpenAI and LangGraph")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("❌ Please set your OPENAI_API_KEY in the .env file")
        st.info("Add your OpenAI API key to the .env file: OPENAI_API_KEY=your_key_here")
        st.stop()
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Resume")
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_pdf")
        resume_text = ""
        if uploaded_resume:
            resume_text = extract_text_from_pdf(uploaded_resume)
            st.success(f"✅ Resume uploaded ({len(resume_text)} characters)")
        else:
            resume_text = st.text_area("Or paste resume text:", height=300, placeholder="Paste your resume content here...")
    
    with col2:
        st.subheader("📑 Job Description")
        uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd_pdf")
        jd_text = ""
        if uploaded_jd:
            jd_text = extract_text_from_pdf(uploaded_jd)
            st.success(f"✅ Job description uploaded ({len(jd_text)} characters)")
        else:
            jd_text = st.text_area("Or paste job description:", height=300, placeholder="Paste the job description here...")
    
    st.markdown("---")
    

    if resume_text and jd_text:
        col3, col4, col5 = st.columns(3)
    
        if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
                run_workflow("matching", resume_text, jd_text)
        
        if st.button("📝 Enhance Resume", use_container_width=True):
                run_workflow("enhancement", resume_text, jd_text)
        
        if st.button("✉️ Generate Cover Letter", use_container_width=True):
                run_workflow("cover_letter", resume_text, jd_text)
    else:
        st.info("👆 Please provide both resume and job description to proceed")

def run_workflow(workflow_type: str, resume_text: str, jd_text: str):
    """Run the specified workflow"""
    try:
        # Initialize state
        initial_state = {
            "resume_text": resume_text,
            "jd_text": jd_text,
            "current_step": "started",
            "errors": [],
            "workflow_type": workflow_type
        }
        
        # Select appropriate workflow
        if workflow_type == "matching":
            workflow = create_matching_workflow()
            spinner_text = "Running match analysis..."
        elif workflow_type == "enhancement":
            workflow = create_enhancement_workflow()
            spinner_text = "Generating resume enhancements..."
        elif workflow_type == "cover_letter":
            workflow = create_cover_letter_workflow()
            spinner_text = "Generating cover letter..."
        else:
            st.error("Invalid workflow type")
            return
        
        # Execute workflow
        with st.spinner(spinner_text):
            result = workflow.invoke(initial_state)
        
        # Display results
        display_results(result, workflow_type)
        
    except Exception as e:
        st.error(f"❌ Workflow execution failed: {str(e)}")
        st.error("Please check your LLM configuration and try again.")

def display_results(result: Dict[str, Any], workflow_type: str):
    """Display workflow results"""
    # Show errors if any
    if result.get("errors"):
        st.error("❌ Analysis completed with errors:")
        for error in result["errors"]:
            st.error(error)
        return
    
    st.success("✅ Analysis Complete!")
    
    # Match Analysis (common to all workflows)
    if "match_analysis" in result:
        match = result["match_analysis"]
        
        # Handle both dictionary and Pydantic model formats
        def get_attr(obj, attr):
            if isinstance(obj, dict):
                return obj.get(attr, [])
            else:
                return getattr(obj, attr, [])
        
        # Create two-column layout
        left_col, right_col = st.columns([1, 1])
        
        # LEFT COLUMN: Match Analysis Results
        with left_col:
            st.markdown("#### 📊 Match Analysis Results")
            
            # Display scores with better formatting
            overall_score = get_attr(match, 'overall_score')
            skills_score = get_attr(match, 'skill_match_score')
            experience_score = get_attr(match, 'experience_match_score')
            
            st.markdown(f"**Overall Match:** {overall_score:.1%}" if overall_score else "**Overall Match:** N/A")
            if overall_score >= 0.8:
                st.markdown("<small>🟢 Excellent Match - Highly Recommended</small>", unsafe_allow_html=True)
            elif overall_score >= 0.6:
                st.markdown("<small>🟡 Good Match - Recommended</small>", unsafe_allow_html=True)
            elif overall_score >= 0.4:
                st.markdown("<small>🟠 Moderate Match - Consider</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small>🔴 Poor Match - Not Recommended</small>", unsafe_allow_html=True)
            
            st.markdown(f"**Skills Match:** {skills_score:.1%}" if skills_score else "**Skills Match:** N/A")
            st.markdown(f"**Experience Match:** {experience_score:.1%}" if experience_score else "**Experience Match:** N/A")
            
            seniority = get_attr(match, 'seniority_assessment')
            if seniority:
                st.markdown(f"**📈 Seniority Level:** <small>{seniority}</small>", unsafe_allow_html=True)
            
            # Strengths and Weaknesses in left column
            st.markdown("---")
            st.markdown("**💡 Assessment Summary**")
            
            strengths = get_attr(match, 'strengths')
            if strengths:
                st.markdown("**💪 Candidate Strengths:**")
                for i, strength in enumerate(strengths, 1):
                    st.markdown(f"<small>✅ {i}. {strength}</small>", unsafe_allow_html=True)
            
            weaknesses = get_attr(match, 'weaknesses')
            if weaknesses:
                st.markdown("**⚠️ Areas for Improvement:**")
                for i, weakness in enumerate(weaknesses, 1):
                    st.markdown(f"<small>❌ {i}. {weakness}</small>", unsafe_allow_html=True)
        
        # RIGHT COLUMN: Skills Analysis
        with right_col:
            st.markdown("#### 🔍 Skills Analysis")
            
            matched_skills = get_attr(match, 'matched_skills')
            missing_skills = get_attr(match, 'missing_skills')
            
            if matched_skills:
                st.markdown("**✅ Matched Skills**")
                for skill in matched_skills:
                    st.markdown(f"<small>✓ {skill}</small>", unsafe_allow_html=True)
                st.markdown("")
            
            if missing_skills:
                st.markdown("**❌ Missing Skills**")
                for skill in missing_skills:
                    st.markdown(f"<small>✗ {skill}</small>", unsafe_allow_html=True)
                st.markdown("")
            
            # Add recommendations to right column
            recommendations = get_attr(match, 'recommendations')
            if recommendations:
                st.markdown("---")
                st.markdown("**🎯 Recommendations**")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"<small>{i}. {rec}</small>", unsafe_allow_html=True)
            
            # Risk factors in right column
            risk_factors = get_attr(match, 'risk_factors')
            if risk_factors:
                st.markdown("---")
                st.markdown("**⚠️ Risk Factors**")
                for i, risk in enumerate(risk_factors, 1):
                    st.markdown(f"<small>{i}. {risk}</small>", unsafe_allow_html=True)
    
    # Enhancement-specific results
    if workflow_type == "enhancement" and "resume_enhancement" in result:
        st.markdown("---")
        st.subheader("📝 Resume Enhancement Suggestions")
        enhancement = result["resume_enhancement"]
        
        # Handle both dictionary and Pydantic model formats
        def get_enhancement_attr(obj, attr):
            if isinstance(obj, dict):
                return obj.get(attr, [])
            else:
                return getattr(obj, attr, [])
        
        # Summary Improvements - Full width
        summary_improvements = get_enhancement_attr(enhancement, 'summary_improvements')
        if summary_improvements:
            st.markdown("**📋 Summary Improvements:**")
            for i, improvement in enumerate(summary_improvements, 1):
                st.info(f"{i}. {improvement}")
            st.markdown("")
        
        # Skill Additions - Full width
        skill_additions = get_enhancement_attr(enhancement, 'skill_additions')
        if skill_additions:
            st.markdown("**🎯 Skills to Add or Emphasize:**")
            skills_text = " • ".join(skill_additions)
            st.success(f"• {skills_text}")
            st.markdown("")
        
        # Keyword Optimization - Full width
        keyword_optimization = get_enhancement_attr(enhancement, 'keyword_optimization')
        if keyword_optimization:
            st.markdown("**🔍 ATS Keyword Optimization:**")
            keywords_text = " • ".join(keyword_optimization)
            st.warning(f"• {keywords_text}")
            st.markdown("")
        
        # Formatting Suggestions - Full width
        formatting_suggestions = get_enhancement_attr(enhancement, 'formatting_suggestions')
        if formatting_suggestions:
            st.markdown("**📐 Formatting Suggestions:**")
            for i, suggestion in enumerate(formatting_suggestions, 1):
                st.write(f"{i}. {suggestion}")
            st.markdown("")
        
        # Overall Recommendations - Full width
        overall_recommendations = get_enhancement_attr(enhancement, 'overall_recommendations')
        if overall_recommendations:
            st.markdown("**🎯 Overall Enhancement Recommendations:**")
            for i, rec in enumerate(overall_recommendations, 1):
                st.write(f"{i}. {rec}")
    
    # Cover letter-specific results
    if workflow_type == "cover_letter" and "cover_letter" in result:
        st.subheader("✉️ Generated Cover Letter")
        cover_letter = result["cover_letter"]
        
        st.text_area("Complete Cover Letter", cover_letter.full_letter, height=400)
        
        # Download option
        st.download_button(
            label="📥 Download Cover Letter",
            data=cover_letter.full_letter,
            file_name="cover_letter.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
