# 🧠 Resume-JD Matcher

An AI-powered resume analysis and job description matching application built with Streamlit, LangGraph, and OpenAI's GPT models. This application helps job seekers analyze how well their resume matches a specific job description and provides actionable insights for improvement.

## ✨ Features

- **📊 Resume-JD Matching Analysis**: Get comprehensive match scores and detailed analysis
- **📝 Resume Enhancement Suggestions**: Receive AI-powered recommendations to improve your resume
- **✉️ Cover Letter Generation**: Generate tailored cover letters based on your resume and job description
- **📄 PDF Support**: Upload resume and job descriptions as PDF files or paste text directly
- **🎯 Skills Gap Analysis**: Identify matched and missing skills
- **⚠️ Risk Assessment**: Get insights on potential hiring concerns
- **📈 Seniority Assessment**: Understand how your experience level aligns with the job requirements

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MultiAgents-with-CrewAI-ResumeJDMatcher
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

4. **Run the application**
   ```bash
   streamlit run unified_langgraph_resume_matcher.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501` to access the application.

## 📦 Dependencies

```
streamlit
langchain-openai
langgraph
PyPDF2
python-dotenv
pydantic
```

## 🎯 How to Use

### 1. Input Your Data
- **Resume**: Upload a PDF file or paste your resume text
- **Job Description**: Upload a PDF file or paste the job description

### 2. Choose Your Analysis
- **🚀 Analyze Match**: Get a comprehensive matching analysis
- **📝 Enhance Resume**: Receive detailed improvement suggestions
- **✉️ Generate Cover Letter**: Create a tailored cover letter

### 3. Review Results
The application provides results in an organized two-column layout:

**Left Column - Match Analysis:**
- Overall match score with color-coded recommendations
- Skills and experience match percentages
- Seniority level assessment
- Candidate strengths and areas for improvement

**Right Column - Skills Analysis:**
- Matched skills with checkmarks
- Missing skills with detailed gaps
- Actionable recommendations
- Risk factors to consider

## 🏗️ Architecture

The application uses a multi-agent architecture powered by LangGraph:

### Agents
1. **Resume Parsing Agent**: Extracts structured data from resumes
2. **Job Description Understanding Agent**: Analyzes job requirements
3. **Matching Agent**: Computes compatibility scores and analysis
4. **Resume Enhancer Agent**: Provides improvement suggestions
5. **Cover Letter Agent**: Generates personalized cover letters

### Workflows
- **Matching Workflow**: Resume → JD Analysis → Match Analysis
- **Enhancement Workflow**: Resume → JD Analysis → Match Analysis → Enhancement Suggestions
- **Cover Letter Workflow**: Resume → JD Analysis → Match Analysis → Cover Letter Generation

### Data Models
The application uses Pydantic models for structured data:
- `ResumeData`: Candidate information and experience
- `JobDescriptionData`: Job requirements and specifications
- `MatchAnalysis`: Comprehensive matching results
- `ResumeEnhancement`: Improvement recommendations
- `CoverLetter`: Generated cover letter content

## 🎨 User Interface

- **Wide Layout**: Optimized for desktop viewing
- **Two-Column Results**: Organized display of analysis results
- **Color-Coded Feedback**: Visual indicators for match quality
- **Interactive Elements**: File uploads, text areas, and action buttons
- **Compact Text**: Optimized font sizes for better readability

## 📊 Scoring System

### Overall Match Score
- **90-100%**: 🟢 Excellent Match - Highly Recommended
- **70-89%**: 🟡 Good Match - Recommended
- **40-69%**: 🟠 Moderate Match - Consider with reservations
- **0-39%**: 🔴 Poor Match - Not recommended

### Analysis Components
- **Skills Assessment**: Technical and soft skills comparison
- **Experience Evaluation**: Relevance and seniority matching
- **Gap Analysis**: Missing requirements identification
- **Risk Assessment**: Potential hiring concerns
- **Recommendations**: Actionable improvement advice

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4o-mini)

### Model Settings
- Temperature: 0.1 (for consistent results)
- Max Tokens: 4000 (for comprehensive analysis)

## 📝 Example Usage

1. **Upload Your Resume**: PDF or text format
2. **Add Job Description**: Copy-paste or upload the JD
3. **Click "Analyze Match"**: Get instant compatibility analysis
4. **Review Results**: 
   - Match scores and recommendations
   - Skills gaps and strengths
   - Improvement suggestions
5. **Optional**: Generate enhanced resume suggestions or cover letter

## 🛠️ Troubleshooting

### Common Issues

**API Key Error**
```
❌ Please set your OPENAI_API_KEY in the .env file
```
- Solution: Add your OpenAI API key to the `.env` file

**PDF Reading Error**
- Ensure your PDF is text-based (not scanned images)
- Try copying and pasting the text instead

**Analysis Timeout**
- Large documents may take longer to process
- Consider breaking down very long resumes or job descriptions

**Poor Match Results**
- Ensure the resume and JD are in the same language
- Check that the job description is complete and detailed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain & LangGraph**: For the multi-agent framework
- **Streamlit**: For the interactive web interface
- **OpenAI**: For the powerful language models
- **Pydantic**: For data validation and parsing

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the logs in the terminal
3. Ensure all dependencies are correctly installed
4. Verify your OpenAI API key is valid and has sufficient credits

## 🔮 Future Enhancements

- [ ] Support for multiple resume formats (DOCX, TXT)
- [ ] Batch processing for multiple resumes
- [ ] Integration with job boards APIs
- [ ] Resume scoring history and tracking
- [ ] Custom scoring weights and preferences
- [ ] Multi-language support
- [ ] Export results to PDF/Word

---

**Happy Job Hunting! 🎯**
