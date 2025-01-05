from django.core.management.base import BaseCommand
from news_app.models import NewsCategory, TrainedData
from datasets import load_dataset
from tqdm import tqdm


class Command(BaseCommand):
    help = 'Import AGNews dataset into TrainedData model'

    def handle(self, *args, **kwargs):
        # Clear existing data
        self.stdout.write("Clearing existing data...")
        TrainedData.objects.all().delete()
        NewsCategory.objects.all().delete()

        # Load AGNews dataset
        dataset = load_dataset("ag_news")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Predefined category mapping
        category_mapping = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Science/Tech"
        }

        # Create or get categories
        categories = {}
        for idx, name in category_mapping.items():
            category = NewsCategory.objects.create(name=name)
            categories[idx] = category
            self.stdout.write(f"Created category: {name}")

        # Bulk create TrainedData objects
        batch_size = 1000
        trained_data_objects = []

        self.stdout.write("Starting import training dataset...")

        # Process training set
        for i in tqdm(range(len(train_dataset)), desc="Processing training data"):
            item = train_dataset[i]
            category = categories[item['label']]

            trained_data = TrainedData(
                title=f"Training Data {i}",
                content=item['text'],
                category=category
            )
            trained_data_objects.append(trained_data)

            if len(trained_data_objects) >= batch_size:
                TrainedData.objects.bulk_create(trained_data_objects)
                trained_data_objects = []

        # Create remaining objects
        if trained_data_objects:
            TrainedData.objects.bulk_create(trained_data_objects)

        total_count = TrainedData.objects.count()
        self.stdout.write(self.style.SUCCESS(
            f'Successfully imported {total_count} records from AGNews dataset'))
