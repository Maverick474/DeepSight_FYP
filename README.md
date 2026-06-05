# DeepSight: Unmasking Manipulated Media Through AI Powered Precision

## Abstract
Deepfakes are AI-generated synthetic media, typically videos or audio, that manipulate a person’s appearance or voice to create highly realistic but fake content. Created using deep learning techniques like GANs and autoencoders, they can be used for both creative and malicious purposes. While useful in areas like film and education, deepfakes pose serious risks to privacy, security, and digital trust by enabling the spread of misinformation and identity manipulation.

## Overview
DeepSight leverages ResNeXt-50 as its core feature extraction backbone for robust visual pattern recognition. It utilizes LSTM for temporal sequence modeling to analyze frame-to-frame consistency, and by integrating an attention mechanism, it focuses on the most important frames and highlights their critical regions. This combination is all optimized for real-time performance at 40 FPS. Code for the model can be found [here](https://github.com/Maverick474/DeepSight_Model.git)

## Installation and Setup
### Prerequisites
- Python 3.11
- Node.js 14 or Higher
- Git (for windows)
- Winrar

Clone the Repository
`git clone https://github.com/Maverick474/DeepSight_FYP.git`

Extract using winrar

## Backend Server Setup (FastApi)
- Navigate to the folder by the name of server using this command on Git Bash:  
`cd server`

- Create virtual environment:  
`python -m venv venv`

- Activate virtual environment (on Windows using Git Bash):  
`source venv/Scripts/activate`

- Activate virtual environment (on Mac/Linus):  
`source venv/Scripts/activate`

- Install dependencies:  
`pip install -r requirements.txt`

- Create model folder in server directory:  
`mkdir models`

- Run the FastAPI server:  
`uvicorn app:app --reload`


## Frontend Setup (React Web App)
- Open another terminal and navigate to web app directory (from main dir):  
`cd frontend`

- Install node modules:  
`npm i`

- Run the React app:  
`npm run dev`

- Web app is running on port:  
`localhost: http://localhost:5173/`

## Frontend Extension Setup
- Open up google chrome and navigate to page `chrome://extensions/`
- Enable developer mode
- Load the extension from extension folder
- Navigate to socail platforms such as Youtube Twitter Instagram and start detecting videos

## Usage
### Web Interface

1. Open your browser and navigate to `localhost: http://localhost:5173/`
2. Upload a video file through the web interface
3. Wait for the analysis to complete
4. View the deepfake detection results with confidence score

## Chrome Extension
1. Installation: Follow the Browser Extension Setup instructions above
2. Activation: Click the DeepSight extension icon in your browser toolbar
3. Real-time Detection: The extension automatically scans video content on web pages
4. Results: Detection results appear as overlay notifications on detected videos
5. Supported Platforms: Works on YouTube Twitter Instagram, social media platforms, and other video hosting sites

## Demo
### Web App

https://github.com/user-attachments/assets/10ac11b2-f53a-4162-92ea-b9c85cdbae82

### Web Extension

https://github.com/user-attachments/assets/61a0a52f-c2f2-4cf9-b6d0-e870f4350a0f

## Contribution Team
### Students
- M. Zain Haseeb
- Tayyab Imam
- Mahnoor Tareen

### Advisors
- Asma Basharat (Primary)
- Muneeb Rashid

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, please open an issue on GitHub or contact the development team.

## Acknowledgments
The ResNext-50 architecture is designed for robust feature extraction, enabling the model to capture complex patterns in data effectively. LSTM networks are utilized for temporal sequence analysis, providing the model with the ability to understand and predict sequences over time. Additionally, Attention mechanisms are incorporated to enhance detection accuracy by allowing the model to focus on the most relevant parts of the input data.










