from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.utils import timezone
from .models import NewsCategory, TrainedData, UntrainedData, UserProfile
from django.contrib.auth.models import User
from News_prediction.code.ai_algorithm import classifier
from .forms import UserRegistrationForm
from django.contrib.auth import login
import json
from django.core.paginator import Paginator
import os
import shutil
from datetime import datetime
from django.http import HttpResponseRedirect
from django.urls import reverse
from News_prediction.code.AGNews_project import train_model_with_new_data


def home(request):
    """Home page view"""
    return render(request, 'news_app/home.html')


@login_required
def predict_news(request):
    """News prediction view"""
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()

        if not content:
            messages.error(request, 'News content cannot be empty')
            return redirect('predict_news')

        # Make prediction
        category_name, confidence = classifier.predict(content)

        if category_name and confidence:
            # Get or create category
            category, _ = NewsCategory.objects.get_or_create(
                name=category_name)

            # Save untrained data
            UntrainedData.objects.create(
                title=title or None,  # If title is empty string, save as None
                content=content,
                predicted_category=category,
                submitted_by=request.user,
                confidence_score=confidence
            )

            # Update user prediction record
            profile, _ = UserProfile.objects.get_or_create(user=request.user)
            profile.last_prediction = timezone.now()
            profile.prediction_count += 1
            profile.save()

            messages.success(
                request,
                f'Prediction complete! Category: {
                    category_name}, Confidence: {confidence:.2f}%'
            )
        else:
            messages.error(
                request, 'Prediction failed, please try again later')

    return render(request, 'news_app/predict.html')


@staff_member_required
def dashboard(request):
    """Admin dashboard view"""
    # Get statistics
    total_trained = TrainedData.objects.count()
    total_untrained = UntrainedData.objects.count()
    categories = NewsCategory.objects.all()

    # Prepare category statistics
    trained_stats = {}
    untrained_stats = {}

    for category in categories:
        trained_stats[category.name] = TrainedData.objects.filter(
            category=category).count()
        untrained_stats[category.name] = UntrainedData.objects.filter(
            predicted_category=category).count()

    context = {
        'total_trained': total_trained,
        'total_untrained': total_untrained,
        'trained_stats': json.dumps(trained_stats),
        'untrained_stats': json.dumps(untrained_stats),
    }

    return render(request, 'news_app/dashboard.html', context)


@staff_member_required
def manage_data(request):
    """Data management view with pagination and sorting"""
    data_type = request.GET.get('type', 'trained')
    page_number = request.GET.get('page', 1)
    # Default sort by creation time
    sort = request.GET.get('sort', 'created_at')
    order = request.GET.get('order', 'desc')
    items_per_page = 20

    # Determine sort field
    if sort == 'category':
        sort_field = 'category' if data_type == 'trained' else 'predicted_category'
    elif sort == 'confidence' and data_type == 'untrained':
        sort_field = 'confidence_score'
    elif sort == 'created_at':
        sort_field = 'created_at'
    elif sort == 'status' and data_type == 'untrained':
        sort_field = 'is_verified'
    else:
        sort_field = 'created_at'  # Default sort field

    # Add sort direction
    if order == 'desc':
        sort_field = f'-{sort_field}'

    # Get data and apply sorting
    if data_type == 'trained':
        data_list = TrainedData.objects.all()
    else:
        data_list = UntrainedData.objects.all()

    # Apply sorting
    data_list = data_list.order_by(sort_field)

    # Create paginator
    paginator = Paginator(data_list, items_per_page)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'data_type': data_type,
        'total_trained': TrainedData.objects.count(),
        'total_untrained': UntrainedData.objects.count(),
        'sort': sort,
        'order': order,
    }

    return render(request, 'news_app/manage_data.html', context)


def register(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Create user profile
            UserProfile.objects.create(user=user)
            login(request, user)
            messages.success(
                request, 'Registration successful! Welcome to News Classification System.')
            return redirect('home')
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'form': form})


@staff_member_required
def manage_admins(request):
    """Admin management view"""
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        action = request.POST.get('action')

        try:
            user = User.objects.get(id=user_id)
            if action == 'make_admin':
                user.is_staff = True
                user.save()
                messages.success(
                    request, f'{user.username} has been made an administrator')
            elif action == 'remove_admin':
                user.is_staff = False
                user.save()
                messages.success(
                    request, f'Administrator privileges removed from {user.username}')
        except User.DoesNotExist:
            messages.error(request, 'User does not exist')

    # Get all users list
    users = User.objects.all().order_by('-date_joined')
    return render(request, 'news_app/manage_admins.html', {'users': users})


@login_required
def profile_view(request):
    """User profile view"""
    # Get or create user profile
    profile, created = UserProfile.objects.get_or_create(user=request.user)

    context = {
        'profile': profile,
        'prediction_count': profile.prediction_count,
        'last_prediction': profile.last_prediction,
    }

    return render(request, 'news_app/profile.html', context)


@staff_member_required
def train_selected_data(request):
    if request.method == 'POST':
        selected_ids = request.POST.getlist('selected_data[]')
        if selected_ids:
            try:
                # Get selected data
                selected_data = UntrainedData.objects.filter(
                    id__in=selected_ids)

                # Prepare training data
                new_training_data = []
                for data in selected_data:
                    new_training_data.append({
                        "text": data.content,
                        "label": data.predicted_category.name
                    })

                if new_training_data:
                    # Backup current model
                    current_model_path = os.path.join(os.path.dirname(
                        os.path.dirname(__file__)), "News_prediction", "code", "lora_final_model")
                    backup_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "News_prediction", "code", "model_backups",
                                               f"lora_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    print(f"Current model path: {current_model_path}")
                    print(f"Backup path: {backup_path}")

                    if os.path.exists(current_model_path):
                        os.makedirs(os.path.dirname(
                            backup_path), exist_ok=True)
                        shutil.copytree(current_model_path, backup_path)
                        print("Backup created successfully")

                    try:
                        # Train model
                        train_model_with_new_data(new_training_data)

                        # Move selected data to trained data
                        for data in selected_data:
                            TrainedData.objects.create(
                                content=data.content,
                                category=data.predicted_category
                            )
                            # Mark as verified
                            data.is_verified = True
                            data.save()

                        messages.success(request, f'Successfully trained model with {
                                         len(selected_data)} new samples!')
                    except Exception as e:
                        # If training fails, restore backup
                        if os.path.exists(backup_path):
                            shutil.rmtree(current_model_path,
                                          ignore_errors=True)
                            shutil.copytree(backup_path, current_model_path)
                        messages.error(
                            request, f'Error during model training: {str(e)}')
                else:
                    messages.warning(
                        request, 'No valid training data selected.')
            except Exception as e:
                messages.error(
                    request, f'Error processing selected data: {str(e)}')
        else:
            messages.warning(request, 'No data selected for training.')

    return HttpResponseRedirect(reverse('manage_data') + '?type=untrained')
