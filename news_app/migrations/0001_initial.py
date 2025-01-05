# Generated by Django 5.1.4 on 2025-01-04 00:13

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="NewsCategory",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100, unique=True)),
                ("description", models.TextField(blank=True)),
            ],
            options={
                "verbose_name": "News Category",
                "verbose_name_plural": "News Categories",
            },
        ),
        migrations.CreateModel(
            name="TrainedData",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("title", models.CharField(max_length=200, verbose_name="Title")),
                ("content", models.TextField(verbose_name="Content")),
                (
                    "created_at",
                    models.DateTimeField(
                        default=django.utils.timezone.now, verbose_name="Created At"
                    ),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, verbose_name="Updated At"),
                ),
                (
                    "category",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="news_app.newscategory",
                        verbose_name="Category",
                    ),
                ),
            ],
            options={
                "verbose_name": "Trained Data",
                "verbose_name_plural": "Trained Data",
            },
        ),
        migrations.CreateModel(
            name="UntrainedData",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("title", models.CharField(max_length=200, verbose_name="Title")),
                ("content", models.TextField(verbose_name="Content")),
                (
                    "confidence_score",
                    models.FloatField(default=0.0, verbose_name="Confidence Score"),
                ),
                (
                    "created_at",
                    models.DateTimeField(
                        default=django.utils.timezone.now, verbose_name="Created At"
                    ),
                ),
                (
                    "is_verified",
                    models.BooleanField(default=False, verbose_name="Verified"),
                ),
                (
                    "predicted_category",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="news_app.newscategory",
                        verbose_name="Predicted Category",
                    ),
                ),
                (
                    "submitted_by",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                        verbose_name="Submitted By",
                    ),
                ),
            ],
            options={
                "verbose_name": "Untrained Data",
                "verbose_name_plural": "Untrained Data",
            },
        ),
        migrations.CreateModel(
            name="UserProfile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "bio",
                    models.TextField(
                        blank=True, max_length=500, verbose_name="Biography"
                    ),
                ),
                (
                    "last_prediction",
                    models.DateTimeField(
                        blank=True, null=True, verbose_name="Last Prediction"
                    ),
                ),
                (
                    "prediction_count",
                    models.IntegerField(default=0, verbose_name="Prediction Count"),
                ),
                (
                    "user",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "User Profile",
                "verbose_name_plural": "User Profiles",
            },
        ),
    ]
