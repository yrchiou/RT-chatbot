# RT-chatbot

## Project Overview
- **Operating System**: Ubuntu 20.04
- **Development Environment**: Jupyter Lab
- **Agent's Brain**: Accessed through ChatNVIDIA (langchain-nvidia-ai-endpoints 0.0.19) to llama3-70b-instruct
- **Agent**: Established through LangChain 0.2.1 as a ReAct Agent
- **RAG Technology**: Used for retrieving relevant data when encountering professional domain questions, enhancing the accuracy of the answers
- **Memory Function**: ConversationBufferWindowMemory, used to avoid the limitations of the context window
- **Communication Software**: LINE, controlled by line-bot-sdk 2.1.0 for real-time conversation between users and the Agent

## Introduction
This project introduces a health education robot for the Radiation Oncology Department. The need for this robot arises from the desire to provide 24/7 service and to bring healthcare closer to patients.

## Technology
The robot uses the NVIDIA Inference Microservice (NIM) with llama3-70b-instruct as the agent's brain. It integrates Retrieval-Augmented Generation (RAG) technology to acquire professional knowledge related to the hospital.

## Proof-of-Concept
NIM helps to quickly achieve a proof-of-concept, significantly reducing the complexity of the project.

## Interface
The robot uses a commonly used communication software, Line Bot, as an interface to enhance the user experience for patients.

## Conclusion
This project aims to bring healthcare closer to patients by providing a 24/7 service. By integrating advanced technologies and user-friendly interfaces, we hope to make a significant contribution to the field of health education in the Radiation Oncology Department.
