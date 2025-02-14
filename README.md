# RAG-Based Personal Legal Assistant

![Image 1](https://github.com/nishikanta24/RAG/blob/main/pics/Screenshot%202025-02-14%20164809.png)
![Image 2](https://github.com/nishikanta24/RAG/blob/main/pics/Screenshot%202025-02-14%20164827.png)

## Overview
This project is a **Retrieval-Augmented Generation (RAG)**-based legal chatbot designed to assist individuals in navigating the complexities of the Indian justice system. The chatbot leverages a **large language model (LLM)** enriched with comprehensive legal knowledge, including **all Indian court cases, penal codes, and law textbooks**. The goal is to empower individuals by providing them with **clear, structured guidance** when dealing with legal issues, helping them make informed decisions without unnecessary legal hurdles.

## Objectives
- Provide **easy-to-understand legal assistance** for common legal issues.
- Offer step-by-step guidance for users facing legal troubles.
- Reduce dependence on expensive legal consultations for preliminary queries.
- Help users understand their **rights, laws, and judicial processes**.
- Ensure accessibility to legal knowledge for everyone.

## Tech Stack
- **Vector Store**: FAISS
- **LLM**: Hugging Face's Mistral
- **Frontend**: Streamlit
- **Backend**: Python

## Features (Planned & Implemented)
- ✅ **Legal Chatbot** trained on Indian case laws, penal codes, and law textbooks.
- ✅ **Step-by-Step Legal Guidance** for common legal issues.
- ✅ **User Query Understanding** with natural language processing.
- ✅ **Document Search & Retrieval** for relevant legal precedents.
- 🛠️ **User Document Upload** for case-specific analysis *(Planned).*
- 🛠️ **Multi-Language Support** for better accessibility *(Planned).*
- 🛠️ **AI-Powered Legal Recommendations** based on case details *(Planned).*
- 🛠️ **Voice Query Support** for accessibility *(Planned).*

## How It Works
1. **User Inputs Legal Query**: The chatbot takes natural language queries from users.
2. **Retrieval of Relevant Cases**: FAISS-based vector search finds the most relevant legal documents.
3. **LLM Generates Response**: The chatbot provides explanations and legal guidance.
4. **User Follow-Up**: The user can refine queries for deeper insights.

## Future Enhancements
- **Integration with Indian Court Databases** for real-time case law updates.
- **Legal Community Forum** for expert legal discussions.
- **Automated Legal Form Generation** based on user input.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
[MIT License](LICENSE)

