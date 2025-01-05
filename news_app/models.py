from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class NewsCategory(models.Model):
    """News category model"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'News Category'
        verbose_name_plural = 'News Categories'


class TrainedData(models.Model):
    """Model for trained news data"""
    title = models.CharField(max_length=200, verbose_name='Title')
    content = models.TextField(verbose_name='Content')
    category = models.ForeignKey(
        NewsCategory, on_delete=models.CASCADE, verbose_name='Category')
    created_at = models.DateTimeField(
        default=timezone.now, verbose_name='Created At')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Updated At')

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = 'Trained Data'
        verbose_name_plural = 'Trained Data'


class UntrainedData(models.Model):
    """Model for untrained news data"""
    title = models.CharField(max_length=200, blank=True,
                             null=True, verbose_name='Title')
    content = models.TextField(verbose_name='Content')
    predicted_category = models.ForeignKey(
        NewsCategory, on_delete=models.CASCADE, verbose_name='Predicted Category')
    submitted_by = models.ForeignKey(
        User, on_delete=models.CASCADE, verbose_name='Submitted By')
    confidence_score = models.FloatField(
        default=0.0, verbose_name='Confidence Score')
    created_at = models.DateTimeField(
        default=timezone.now, verbose_name='Created At')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='Updated At')
    is_verified = models.BooleanField(default=False, verbose_name='Verified')

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = 'Untrained Data'
        verbose_name_plural = 'Untrained Data'


class UserProfile(models.Model):
    """User profile model"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True,
                           verbose_name='Biography')
    last_prediction = models.DateTimeField(
        null=True, blank=True, verbose_name='Last Prediction')
    prediction_count = models.IntegerField(
        default=0, verbose_name='Prediction Count')

    def __str__(self):
        return self.user.username

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
