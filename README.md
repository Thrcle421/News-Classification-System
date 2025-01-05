# News Classification System

This project is an extension of the final project from UC San Diego's CSE 256 course. In the final project, I compared the performance and efficiency of **LoRA-based fine-tuning** with traditional full-parameter fine-tuning for text classification tasks. Using the **IMDB Dataset** and **AG’s News Dataset**, I explored the capabilities of LoRA and further experimented with LoRA+ in this context. The code for the project is packaged under ucsd-cse256, and the corresponding report is titled "**Fine-Tuning for Text Classification using Low-Rank Adaptation (LoRA)**".

Building on this exploration, I developed a **Django-based web application** for news text classification. The application utilizes the LoRA model fine-tuned on the AG’s News Dataset to predict the category of news articles entered by users. This implementation bridges theoretical research with a practical application for real-world use cases.

I also tried using cursor effects to enhance the frontend design.

## Features

### News Classification
- Real-time news text classification using LoRA fine-tuning
- Confidence score for each prediction
- Support for multiple news categories

### Data Management
- Separate management for trained and untrained data
- High confidence data filtering (>90%)
- Advanced sorting and filtering options
- Status tracking for data verification

### User System
- User authentication and authorization
- Staff-only access to training features
- User prediction history tracking
- Profile management

### Dashboard
- Visual statistics with Chart.js
- Category distribution visualization
- Data status overview
- Real-time metrics

### Model Management
- Automatic model backup before training
- LoRA fine-tuning integration
- Model version control
- Training history tracking

## Technical Stack

- **Backend**: Django
- **Frontend**: Bootstrap 5, Chart.js
- **Database**: MySQL
- **ML Model**: LoRA fine-tuning

