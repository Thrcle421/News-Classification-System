# Generated by Django 5.1.4 on 2025-01-04 07:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("news_app", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="untraineddata",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, verbose_name="Updated At"),
        ),
        migrations.AlterField(
            model_name="untraineddata",
            name="title",
            field=models.CharField(
                blank=True, max_length=200, null=True, verbose_name="Title"
            ),
        ),
    ]
