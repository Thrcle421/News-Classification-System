# News Classification System

A Django-based web application for news text classification with fine-tuning capabilities.

## Features

### News Classification
- Real-time news text classification using LoRA fine-tuning
- Confidence score for each prediction
- Support for multiple news categories

### Data Management
- Separate management for trained and untrained data
- Batch selection and training functionality
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
- **Database**: SQLite/PostgreSQL
- **ML Model**: LoRA fine-tuning
- **Authentication**: Django built-in auth

## Key Components

### Models
- NewsCategory: News category management
- TrainedData: Verified and trained news data
- UntrainedData: Pending verification news data
- UserProfile: Extended user information

### Views
- Prediction interface
- Data management dashboard
- Training control panel
- User management
